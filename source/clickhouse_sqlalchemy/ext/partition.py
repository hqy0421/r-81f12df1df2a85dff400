"""Partition publication from a frozen MergeTree staging table.

A monthly recalculation produces a frozen MergeTree staging table. This
module publishes a single staging partition into the production table in
one native ``ALTER TABLE ... REPLACE PARTITION ... FROM ...`` statement.

The workflow is:

1. :func:`build_publish_plan` creates an immutable, serializable
   :class:`PartitionPublishPlan`.
2. :func:`precheck_publish` performs strictly read-only checks: both tables
   live in the same database, are MergeTree-family tables, have identical
   column definitions, partition keys and sorting keys, and the staging
   table contains an active partition with the requested id.
3. :func:`publish_partition` runs the precheck and then executes exactly one
   ``REPLACE PARTITION`` statement. It never drops the old partition first,
   never touches other partitions and never modifies the staging table.

``REPLACE PARTITION`` is atomic and idempotent: re-executing the same plan
after a connection failure replaces the partition again with the very same
data instead of accumulating rows. If the connection breaks while the
statement is in flight and the outcome cannot be confirmed,
:class:`~clickhouse_sqlalchemy.exceptions.PartitionPublishUnknownError`
keeps the original plan, which can be retried as-is (or resumed via
:func:`resume_publish`, which skips the statement when verification shows
that the first attempt already succeeded).
"""

from dataclasses import dataclass, field, asdict
from enum import Enum

from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, InterfaceError, OperationalError

from ..exceptions import (
    DatabaseException, PartitionPrecheckError, PartitionPublishUnknownError
)
from ..sql.ddl import ReplacePartition


# Engine names reported by system.tables for the MergeTree family.
# Replicated variants are spelled ``Replicated*MergeTree``; every member
# ends with the ``MergeTree`` suffix.
_MERGETREE_SUFFIX = 'MergeTree'


class PublishStatus(str, Enum):
    """Result of verifying a partition publication against the server."""

    #: Production partition matches the staging partition (publication
    #: succeeded, possibly during an earlier attempt).
    PUBLISHED = 'published'

    #: Production partition does not match the staging partition. The
    #: statement should be (re-)executed with the same plan.
    NOT_PUBLISHED = 'not_published'

    #: The staging partition used as the publication source is gone, so the
    #: plan cannot be verified or retried.
    STAGING_PARTITION_MISSING = 'staging_partition_missing'


@dataclass(frozen=True)
class PartitionPublishPlan:
    """Immutable description of one partition publication.

    Plans are serializable (:meth:`to_dict` / :meth:`from_dict`) so that a
    plan created before a connection failure can be reused from another
    process or session.
    """

    target_table: str
    source_table: str
    partition_id: str
    database: str = None

    def __post_init__(self):
        if not isinstance(self.target_table, str) or not self.target_table:
            raise ValueError('target_table must be a non-empty string')
        if not isinstance(self.source_table, str) or not self.source_table:
            raise ValueError('source_table must be a non-empty string')
        if self.target_table == self.source_table:
            raise ValueError(
                'target_table and source_table must be different tables'
            )
        # Partition ids are always strings in system.parts, including
        # integer partitions; keep the canonical string form.
        object.__setattr__(self, 'partition_id', str(self.partition_id))
        if not self.partition_id:
            raise ValueError('partition_id must be a non-empty value')
        if self.database is not None and not isinstance(self.database, str):
            raise ValueError('database must be a string or None')

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

    def to_statement(self):
        """Compile into a :class:`ReplacePartition` DDL clause."""
        return ReplacePartition(
            target_table=self.target_table,
            source_table=self.source_table,
            partition_id=self.partition_id,
            target_schema=self.database,
            source_schema=self.database,
        )


@dataclass(frozen=True)
class PrecheckReport:
    """Outcome of the read-only publication precheck."""

    plan: PartitionPublishPlan
    ok: bool
    failures: tuple = field(default_factory=tuple)
    engine: dict = None
    columns: tuple = None
    partition: dict = None

    def raise_if_failed(self):
        if not self.ok:
            raise PartitionPrecheckError(list(self.failures))


def build_publish_plan(target_table, source_table, partition_id,
                       database=None):
    """Create a :class:`PartitionPublishPlan` without touching the server."""
    return PartitionPublishPlan(
        target_table=target_table,
        source_table=source_table,
        partition_id=partition_id,
        database=database,
    )


def _execute(connection, sql, params=None):
    return connection.execute(text(sql), params or {})


def _current_database(connection):
    return _execute(
        connection, 'SELECT currentDatabase() AS db'
    ).scalar_one()


def _fetch_table_rows(connection, database, names):
    rows = _execute(
        connection,
        'SELECT name, engine, partition_key, sorting_key, primary_key '
        'FROM system.tables '
        'WHERE database = :database AND name IN :names',
        {'database': database, 'names': tuple(names)}
    ).mappings().all()
    return {row['name']: dict(row) for row in rows}


def _fetch_columns(connection, database, table):
    rows = _execute(
        connection,
        'DESCRIBE TABLE {database}.{table}'.format(
            database=_quote_identifier(connection, database),
            table=_quote_identifier(connection, table),
        )
    ).mappings().all()
    return tuple(
        (row['name'], row['type']) for row in rows
    )


def _quote_identifier(connection, identifier):
    return connection.dialect.identifier_preparer.quote_identifier(identifier)


def _is_mergetree(engine_name):
    return bool(engine_name) and engine_name.endswith(_MERGETREE_SUFFIX)


def _check_columns_match(staging_columns, target_columns):
    if staging_columns == target_columns:
        return None

    staging_names = [name for name, _ in staging_columns]
    target_names = [name for name, _ in target_columns]

    if staging_names != target_names:
        return (
            'column definitions do not match: staging columns {staging} '
            'differ from target columns {target}'.format(
                staging=staging_names, target=target_names
            )
        )

    mismatches = []
    for (staging_name, staging_type), (target_name, target_type) in zip(
        staging_columns, target_columns
    ):
        if staging_type != target_type:
            mismatches.append(
                '{name}: staging {staging} vs target {target}'.format(
                    name=staging_name,
                    staging=staging_type,
                    target=target_type,
                )
            )
    return 'column types do not match: {}'.format('; '.join(mismatches))


def precheck_publish(connection, plan):
    """Run all read-only checks required before publication.

    The check only reads ``system.tables``, ``system.parts`` and
    ``DESCRIBE TABLE``. It returns a :class:`PrecheckReport`; pass
    ``report.raise_if_failed()`` or set ``raise_on_failure=True`` on the
    publishing helpers to turn failures into
    :class:`~clickhouse_sqlalchemy.exceptions.PartitionPrecheckError`.
    """
    if not isinstance(plan, PartitionPublishPlan):
        raise TypeError(
            'plan must be a PartitionPublishPlan instance, got {!r}'.format(
                type(plan)
            )
        )

    database = plan.database or _current_database(connection)
    failures = []

    tables = _fetch_table_rows(
        connection, database, (plan.source_table, plan.target_table)
    )

    staging = tables.get(plan.source_table)
    target = tables.get(plan.target_table)

    if staging is None:
        failures.append(
            'staging table {}.{} does not exist'.format(
                database, plan.source_table
            )
        )
    if target is None:
        failures.append(
            'target table {}.{} does not exist'.format(
                database, plan.target_table
            )
        )

    engine_info = None
    staging_columns = target_columns = None
    partition_info = None

    if staging is not None and target is not None:
        # Same-database guarantee: rows come from one single database filter.
        engine_info = {
            'source_engine': staging['engine'],
            'target_engine': target['engine'],
        }
        if not _is_mergetree(staging['engine']):
            failures.append(
                'staging table engine {!r} is not a MergeTree-family '
                'engine'.format(staging['engine'])
            )
        if not _is_mergetree(target['engine']):
            failures.append(
                'target table engine {!r} is not a MergeTree-family '
                'engine'.format(target['engine'])
            )

        for key, label in (
            ('partition_key', 'partition key'),
            ('sorting_key', 'sorting key'),
        ):
            if staging[key] != target[key]:
                failures.append(
                    '{label} does not match: staging {staging!r} vs '
                    'target {target!r}'.format(
                        label=label,
                        staging=staging[key],
                        target=target[key],
                    )
                )

        staging_columns = _fetch_columns(
            connection, database, plan.source_table
        )
        target_columns = _fetch_columns(
            connection, database, plan.target_table
        )
        column_failure = _check_columns_match(
            staging_columns, target_columns
        )
        if column_failure is not None:
            failures.append(column_failure)

        # Mandatory: REPLACE PARTITION with a missing source partition
        # silently takes the matching target partition offline, so an
        # active staging partition is a hard requirement.
        partition_info = _fetch_partition_totals(
            connection, database, plan.source_table, plan.partition_id
        )
        if not partition_info['part_count']:
            failures.append(
                'staging table {}.{} has no active partition with '
                'id {!r}'.format(
                    database, plan.source_table, plan.partition_id
                )
            )

    return PrecheckReport(
        plan=plan,
        ok=not failures,
        failures=tuple(failures),
        engine=engine_info,
        columns=(staging_columns, target_columns),
        partition=partition_info,
    )


def verify_publish(connection, plan):
    """Check the publication state without modifying anything.

    Compares a data-level signature (row count plus a ``cityHash64``
    checksum over every column) of the target partition with the staging
    partition, filtered through the ``_partition_id`` virtual column.
    Unlike part names and on-disk bytes, this signature stays stable when
    ClickHouse merges parts in the target table after publication. On
    servers that do not expose ``_partition_id`` it falls back to
    comparing ``system.parts`` totals. Returns a :class:`PublishStatus`.
    """
    database = plan.database or _current_database(connection)

    if _supports_partition_id_virtual_column(connection):
        staging = _fetch_partition_signature(
            connection, database, plan.source_table, plan.partition_id
        )
        if staging['row_count'] == 0:
            return PublishStatus.STAGING_PARTITION_MISSING
        target = _fetch_partition_signature(
            connection, database, plan.target_table, plan.partition_id
        )
    else:
        # Older servers do not expose the _partition_id virtual column;
        # compare system.parts totals instead.
        staging = _fetch_partition_totals(
            connection, database, plan.source_table, plan.partition_id
        )
        if not staging['part_count']:
            return PublishStatus.STAGING_PARTITION_MISSING
        target = _fetch_partition_totals(
            connection, database, plan.target_table, plan.partition_id
        )

    if target == staging:
        return PublishStatus.PUBLISHED
    return PublishStatus.NOT_PUBLISHED


def _supports_partition_id_virtual_column(connection):
    # _partition_id has been available since ClickHouse 18.16.
    version_info = getattr(
        connection.dialect, 'server_version_info', None
    )
    if not version_info:
        return True
    try:
        return tuple(version_info[:3]) >= (18, 16)
    except TypeError:
        return True


def _fetch_partition_signature(connection, database, table, partition_id):
    # _partition_id is always a string, even for integer partition keys.
    # cityHash64(*) covers every column; its sum wraps at 64 bits, which is
    # fine for equality between two identically typed tables.
    quoted_database = _quote_identifier(connection, database)
    quoted_table = _quote_identifier(connection, table)
    rows = _execute(
        connection,
        'SELECT count() AS row_count, '
        'sum(cityHash64(*)) AS row_hash '
        'FROM {database}.{table} '
        'WHERE _partition_id = :partition'.format(
            database=quoted_database, table=quoted_table
        ),
        {'partition': partition_id}
    ).mappings().all()
    row = rows[0]
    return {
        'row_count': row['row_count'],
        'row_hash': row['row_hash'],
    }


def _fetch_partition_totals(connection, database, table, partition_id):
    rows = _execute(
        connection,
        'SELECT count() AS part_count, sum(rows) AS total_rows, '
        'sum(bytes_on_disk) AS total_bytes '
        'FROM system.parts '
        'WHERE database = :database AND table = :table '
        'AND active AND partition = :partition',
        {
            'database': database,
            'table': table,
            'partition': partition_id,
        }
    ).mappings().all()
    row = rows[0]
    return {
        'part_count': row['part_count'],
        'total_rows': row['total_rows'],
        'total_bytes': row['total_bytes'],
    }


def _is_connection_failure(error):
    # Communication failures surface through up to three layers depending
    # on the driver and where the socket breaks:
    #   1. SQLAlchemy wrappers (OperationalError/InterfaceError/DBAPIError);
    #   2. the dialect's own DatabaseException wrapping a driver error;
    #   3. a raw built-in connection error that the driver did not wrap.
    # Statement/syntax errors are deterministic and must propagate.
    if isinstance(error, (OperationalError, InterfaceError)):
        return True
    if isinstance(error, DBAPIError):
        orig = getattr(error, 'orig', None)
        if _is_network_origin(orig):
            return True
    if isinstance(error, DatabaseException) and _is_network_origin(
        getattr(error, 'orig', None)
    ):
        return True
    # Raw socket-level errors that escaped driver wrapping, e.g.
    # ConnectionResetError / BrokenPipeError / built-in TimeoutError.
    if isinstance(error, (ConnectionError, TimeoutError)):
        return True
    return False


def _is_network_origin(orig):
    # clickhouse-driver NetworkError/SocketTimeoutError, the HTTP
    # transport's HTTPException and the requests exception hierarchy.
    return type(orig).__name__ in (
        'NetworkError', 'SocketTimeoutError', 'ConnectionError',
        'Timeout', 'RequestException', 'HTTPException',
    )


def publish_partition(connection, plan, raise_on_failure=True):
    """Publish one staging partition into the target table.

    Runs :func:`precheck_publish` first and then executes exactly one
    native ``REPLACE PARTITION`` statement. The staging table is not
    modified, no partition is dropped beforehand, and no other partition
    is touched.

    Raises :class:`PartitionPublishUnknownError` if the connection breaks
    before the statement result is confirmed. The plan remains valid and
    retrying it is safe and non-accumulating.
    """
    report = precheck_publish(connection, plan)
    if not report.ok:
        if raise_on_failure:
            report.raise_if_failed()
        return report

    statement = plan.to_statement()
    try:
        connection.execute(statement)
    except Exception as error:
        if _is_connection_failure(error):
            raise PartitionPublishUnknownError(
                'connection lost while publishing partition {!r} from '
                '{source} to {target}; the server outcome is unknown, '
                'retry the same plan'.format(
                    plan.partition_id,
                    source=plan.source_table,
                    target=plan.target_table,
                ),
                error,
                plan,
            )
        raise

    return report


def resume_publish(connection, plan, raise_on_failure=True):
    """Retry a plan whose previous outcome is unknown.

    Verifies the current state first. If the first attempt already
    published the partition, no statement is executed. Otherwise the same
    plan is published normally (precheck plus one ``REPLACE PARTITION``).
    """
    status = verify_publish(connection, plan)
    if status is PublishStatus.PUBLISHED:
        return status
    if status is PublishStatus.STAGING_PARTITION_MISSING:
        if raise_on_failure:
            raise PartitionPrecheckError(
                'staging partition {!r} no longer exists; the plan cannot '
                'be retried'.format(plan.partition_id)
            )
        return status
    publish_partition(connection, plan, raise_on_failure=raise_on_failure)
    return PublishStatus.PUBLISHED


def list_active_partitions(connection, table, database=None):
    """Return active partition ids of a MergeTree table.

    Convenient for discovering the exact id to pass to
    :func:`build_publish_plan`; read-only.
    """
    database = database or _current_database(connection)
    rows = _execute(
        connection,
        'SELECT DISTINCT partition '
        'FROM system.parts '
        'WHERE database = :database AND table = :table AND active '
        'ORDER BY partition',
        {'database': database, 'table': table}
    ).all()
    return [row[0] for row in rows]
