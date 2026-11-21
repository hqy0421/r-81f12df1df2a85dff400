from unittest.mock import patch

from sqlalchemy import text
from sqlalchemy.exc import (
    DBAPIError, InterfaceError, OperationalError, ProgrammingError,
)

from clickhouse_driver.errors import (
    NetworkError, ServerException, SocketTimeoutError
)

from clickhouse_sqlalchemy.exceptions import (
    DatabaseException, PartitionPrecheckError, PartitionPublishUnknownError
)
from clickhouse_sqlalchemy.ext import partition as pmod
from clickhouse_sqlalchemy.ext.partition import (
    PartitionPublishPlan, PublishStatus,
    build_publish_plan, precheck_publish, publish_partition,
    resume_publish, verify_publish, list_active_partitions,
)
from tests.testcase import BaseTestCase
from tests.util import with_native_and_http_sessions


STAGING_DDL = (
    'CREATE TABLE {staging} ('
    'id UInt32, d Date, v String'
    ') ENGINE = MergeTree '
    'PARTITION BY toYYYYMM(d) ORDER BY id'
)
TARGET_DDL = STAGING_DDL


@with_native_and_http_sessions
class PartitionPublishTestCase(BaseTestCase):

    def _connection(self):
        return self.session.connection()

    def _create_tables(self, staging='pp_staging', target='pp_target',
                       staging_ddl=None, target_ddl=None):
        conn = self._connection()
        conn.execute(text('DROP TABLE IF EXISTS {}'.format(staging)))
        conn.execute(text('DROP TABLE IF EXISTS {}'.format(target)))
        conn.execute(text(
            (staging_ddl or STAGING_DDL).format(staging=staging)
        ))
        conn.execute(text(
            (target_ddl or TARGET_DDL).format(staging=target)
        ))
        return conn

    def _insert_staging(self, conn, staging):
        conn.execute(text(
            'INSERT INTO {table} VALUES '.format(table=staging)
            + "(1, '2026-01-10', 'a'), "
            + "(2, '2026-01-11', 'b'), "
            + "(3, '2026-02-01', 'c')"
        ))

    def _plan(self, target='pp_target', staging='pp_staging',
              partition_id='202601', database='test'):
        return build_publish_plan(target, staging, partition_id, database)

    def test_plan_validation_and_serialization(self):
        plan = build_publish_plan('t', 's', 5, database='db')
        self.assertEqual(plan.partition_id, '5')
        self.assertEqual(
            plan.to_dict(),
            {
                'target_table': 't',
                'source_table': 's',
                'partition_id': '5',
                'database': 'db',
            }
        )
        self.assertEqual(PartitionPublishPlan.from_dict(plan.to_dict()), plan)

        with self.assertRaises(ValueError):
            build_publish_plan('t', 't', '1')
        with self.assertRaises(ValueError):
            build_publish_plan('', 's', '1')
        with self.assertRaises(ValueError):
            build_publish_plan('t', 's', '')

    def test_precheck_succeeds_for_compatible_tables(self):
        conn = self._create_tables()
        self._insert_staging(conn, 'pp_staging')

        report = precheck_publish(conn, self._plan())
        self.assertTrue(report.ok)
        self.assertEqual(report.failures, ())
        self.assertEqual(
            report.engine,
            {'source_engine': 'MergeTree', 'target_engine': 'MergeTree'}
        )
        self.assertEqual(report.partition['part_count'], 1)
        self.assertEqual(report.partition['total_rows'], 2)
        self.assertEqual(report.columns[0], report.columns[1])

    def test_precheck_uses_current_database_when_unspecified(self):
        conn = self._create_tables()
        self._insert_staging(conn, 'pp_staging')

        plan = build_publish_plan('pp_target', 'pp_staging', '202601')
        report = precheck_publish(conn, plan)
        self.assertTrue(report.ok, report.failures)

    def test_precheck_fails_when_table_missing(self):
        conn = self._connection()
        report = precheck_publish(
            conn, self._plan(target='pp_missing_table')
        )
        self.assertFalse(report.ok)
        self.assertEqual(len(report.failures), 1)
        self.assertIn('does not exist', report.failures[0])

        with self.assertRaises(PartitionPrecheckError) as ctx:
            publish_partition(conn, self._plan(target='pp_missing_table'))
        self.assertEqual(ctx.exception.failures, list(report.failures))

    def test_precheck_fails_for_non_mergetree_engine(self):
        conn = self._connection()
        conn.execute(text('DROP TABLE IF EXISTS pp_memory'))
        conn.execute(text(
            'CREATE TABLE pp_memory (id UInt32) ENGINE = Memory()'
        ))
        report = precheck_publish(
            conn, self._plan(staging='pp_memory')
        )
        self.assertFalse(report.ok)
        self.assertTrue(
            any('MergeTree-family engine' in f for f in report.failures)
        )

    def test_precheck_fails_for_column_mismatch(self):
        conn = self._create_tables(staging_ddl=(
            'CREATE TABLE {staging} ('
            'id UInt32, d Date, v UInt64'
            ') ENGINE = MergeTree '
            'PARTITION BY toYYYYMM(d) ORDER BY id'
        ))
        conn.execute(text(
            "INSERT INTO pp_staging VALUES (1, '2026-01-10', 1)"
        ))
        report = precheck_publish(conn, self._plan())
        self.assertFalse(report.ok)
        self.assertTrue(
            any('column types do not match' in f for f in report.failures)
        )

    def test_precheck_fails_for_partition_key_mismatch(self):
        conn = self._create_tables(staging_ddl=(
            'CREATE TABLE {staging} ('
            'id UInt32, d Date, v String'
            ') ENGINE = MergeTree '
            'PARTITION BY d ORDER BY id'
        ))
        conn.execute(text(
            "INSERT INTO pp_staging VALUES (1, '2026-01-10', 'a')"
        ))
        report = precheck_publish(conn, self._plan(partition_id='2026-01-10'))
        self.assertFalse(report.ok)
        self.assertTrue(
            any('partition key does not match' in f for f in report.failures)
        )

    def test_precheck_fails_for_sorting_key_mismatch(self):
        conn = self._create_tables(staging_ddl=(
            'CREATE TABLE {staging} ('
            'id UInt32, d Date, v String'
            ') ENGINE = MergeTree '
            'PARTITION BY toYYYYMM(d) ORDER BY (id, d)'
        ))
        conn.execute(text(
            "INSERT INTO pp_staging VALUES (1, '2026-01-10', 'a')"
        ))
        report = precheck_publish(conn, self._plan())
        self.assertFalse(report.ok)
        self.assertTrue(
            any('sorting key does not match' in f for f in report.failures)
        )

    def test_precheck_fails_without_active_staging_partition(self):
        conn = self._create_tables()
        self._insert_staging(conn, 'pp_staging')

        report = precheck_publish(conn, self._plan(partition_id='209912'))
        self.assertFalse(report.ok)
        self.assertTrue(
            any('no active partition' in f for f in report.failures)
        )

    def test_publish_replaces_only_requested_partition(self):
        conn = self._create_tables()
        self._insert_staging(conn, 'pp_staging')
        # Old data in the partition to replace plus data in another
        # partition that must survive publication.
        conn.execute(text(
            "INSERT INTO pp_target VALUES "
            "(9, '2026-01-05', 'old'), (8, '2026-02-02', 'keep')"
        ))

        publish_partition(conn, self._plan())

        rows = conn.execute(text(
            'SELECT id, v FROM pp_target ORDER BY id'
        )).all()
        self.assertEqual(
            rows, [(1, 'a'), (2, 'b'), (8, 'keep')]
        )
        # Staging table is untouched.
        self.assertEqual(
            conn.execute(text('SELECT count() FROM pp_staging')).scalar_one(),
            3
        )
        self.assertEqual(
            verify_publish(conn, self._plan()), PublishStatus.PUBLISHED
        )

    def test_publish_executes_single_replace_statement(self):
        conn = self._create_tables()
        self._insert_staging(conn, 'pp_staging')

        executed = []
        real_execute = type(conn).execute

        def recorder(self, statement, parameters=None, *args, **kwargs):
            executed.append(
                str(statement.compile(dialect=self.dialect))
            )
            return real_execute(
                self, statement, parameters, *args, **kwargs
            )

        with patch.object(type(conn), 'execute', recorder):
            publish_partition(conn, self._plan())

        replace_statements = [
            s for s in executed if 'REPLACE PARTITION' in s
        ]
        self.assertEqual(len(replace_statements), 1)
        self.assertEqual(
            replace_statements[0].replace('\n', ''),
            'ALTER TABLE test."pp_target" REPLACE PARTITION \'202601\' '
            'FROM test."pp_staging"'
        )
        # No DROP is issued as part of publication.
        self.assertFalse(
            any('DROP' in s for s in executed)
        )

    def test_publish_is_idempotent_on_retry(self):
        conn = self._create_tables()
        self._insert_staging(conn, 'pp_staging')

        plan = self._plan()
        publish_partition(conn, plan)
        publish_partition(conn, plan)

        # Retrying the same plan does not accumulate rows.
        self.assertEqual(
            conn.execute(text('SELECT count() FROM pp_target')).scalar_one(),
            2
        )
        self.assertEqual(
            conn.execute(text(
                'SELECT sum(id) FROM pp_target'
            )).scalar_one(),
            3
        )

    def test_integer_partition_id(self):
        conn = self._connection()
        conn.execute(text('DROP TABLE IF EXISTS pp_int_s'))
        conn.execute(text('DROP TABLE IF EXISTS pp_int_t'))
        conn.execute(text(
            'CREATE TABLE pp_int_s (id UInt32, m UInt32) '
            'ENGINE = MergeTree PARTITION BY m ORDER BY id'
        ))
        conn.execute(text(
            'CREATE TABLE pp_int_t (id UInt32, m UInt32) '
            'ENGINE = MergeTree PARTITION BY m ORDER BY id'
        ))
        conn.execute(text(
            'INSERT INTO pp_int_s VALUES (1, 5), (2, 5)'
        ))

        plan = build_publish_plan('pp_int_t', 'pp_int_s', 5, database='test')
        report = precheck_publish(conn, plan)
        self.assertTrue(report.ok, report.failures)
        publish_partition(conn, plan)
        self.assertEqual(
            conn.execute(text(
                'SELECT id FROM pp_int_t ORDER BY id'
            )).all(),
            [(1,), (2,)]
        )
        self.assertEqual(
            verify_publish(conn, plan), PublishStatus.PUBLISHED
        )

    def test_connection_failure_keeps_plan_and_resume_retries(self):
        conn = self._create_tables()
        self._insert_staging(conn, 'pp_staging')
        plan = self._plan()

        real_execute = type(conn).execute
        state = {'fail_next_replace': True, 'replace_calls': 0}

        def flaky(self, statement, parameters=None, *args, **kwargs):
            from clickhouse_sqlalchemy.sql.ddl import ReplacePartition
            if isinstance(statement, ReplacePartition):
                state['replace_calls'] += 1
                if state['fail_next_replace']:
                    state['fail_next_replace'] = False
                    raise OperationalError(
                        None, {}, NetworkError('connection lost')
                    )
            return real_execute(
                self, statement, parameters, *args, **kwargs
            )

        with patch.object(type(conn), 'execute', flaky):
            with self.assertRaises(PartitionPublishUnknownError) as ctx:
                publish_partition(conn, plan)

        self.assertEqual(ctx.exception.plan, plan)
        self.assertIsInstance(
            ctx.exception.original_error, OperationalError
        )

        # Same plan, retry through resume; staging is still in place.
        status = resume_publish(conn, plan)
        self.assertEqual(status, PublishStatus.PUBLISHED)
        self.assertEqual(state['replace_calls'], 1)
        self.assertEqual(
            conn.execute(text('SELECT count() FROM pp_target')).scalar_one(),
            2
        )

    def test_resume_skips_statement_when_already_published(self):
        conn = self._create_tables()
        self._insert_staging(conn, 'pp_staging')
        plan = self._plan()
        publish_partition(conn, plan)

        real_execute = type(conn).execute

        def fail_on_replace(self, statement, parameters=None, *a, **kw):
            from clickhouse_sqlalchemy.sql.ddl import ReplacePartition
            if isinstance(statement, ReplacePartition):
                raise AssertionError(
                    'REPLACE must not be re-executed once published'
                )
            return real_execute(self, statement, parameters, *a, **kw)

        with patch.object(type(conn), 'execute', fail_on_replace):
            self.assertEqual(
                resume_publish(conn, plan), PublishStatus.PUBLISHED
            )

    def test_verify_reports_not_published_beforehand(self):
        conn = self._create_tables()
        self._insert_staging(conn, 'pp_staging')
        plan = self._plan()
        self.assertEqual(
            verify_publish(conn, plan), PublishStatus.NOT_PUBLISHED
        )

    def test_verify_reports_missing_staging_partition(self):
        conn = self._create_tables()
        plan = self._plan(partition_id='209912')
        self.assertEqual(
            verify_publish(conn, plan),
            PublishStatus.STAGING_PARTITION_MISSING
        )

    def test_verify_stays_published_after_target_merge(self):
        # The data-level signature must remain equal even when ClickHouse
        # merges the target partition's parts (which changes part count and
        # on-disk bytes but not the rows).
        conn = self._create_tables()
        self._insert_staging(conn, 'pp_staging')
        conn.execute(text(
            "INSERT INTO pp_staging VALUES "
            "(4, '2026-01-20', 'd'), (5, '2026-01-21', 'e')"
        ))
        plan = self._plan()
        publish_partition(conn, plan)
        self.assertEqual(
            verify_publish(conn, plan), PublishStatus.PUBLISHED
        )

        conn.execute(text('OPTIMIZE TABLE pp_target FINAL'))
        self.assertEqual(
            verify_publish(conn, plan), PublishStatus.PUBLISHED
        )

    def test_verify_detects_content_difference(self):
        conn = self._create_tables()
        self._insert_staging(conn, 'pp_staging')
        conn.execute(text(
            "INSERT INTO pp_target VALUES (7, '2026-01-07', 'different')"
        ))
        plan = self._plan()
        self.assertEqual(
            verify_publish(conn, plan), PublishStatus.NOT_PUBLISHED
        )

    def test_list_active_partitions(self):
        conn = self._create_tables()
        self._insert_staging(conn, 'pp_staging')
        self.assertEqual(
            list_active_partitions(conn, 'pp_staging', 'test'),
            ['202601', '202602']
        )

    def test_precheck_does_not_modify_anything(self):
        conn = self._create_tables()
        self._insert_staging(conn, 'pp_staging')
        conn.execute(text(
            "INSERT INTO pp_target VALUES (9, '2026-01-05', 'old')"
        ))

        real_execute = type(conn).execute
        mutating = []

        def checker(self, statement, parameters=None, *args, **kwargs):
            sql = str(statement).upper()
            if any(keyword in sql for keyword in (
                'INSERT', 'ALTER', 'DROP', 'CREATE', 'DELETE', 'UPDATE'
            )):
                mutating.append(sql)
            return real_execute(
                self, statement, parameters, *args, **kwargs
            )

        with patch.object(type(conn), 'execute', checker):
            report = precheck_publish(conn, self._plan())

        self.assertTrue(report.ok)
        self.assertEqual(mutating, [])
        # Pre-existing target data unchanged by the precheck.
        self.assertEqual(
            conn.execute(text('SELECT id FROM pp_target')).all(), [(9,)]
        )


class ConnectionFailureClassificationTestCase(BaseTestCase):
    """The classifier decides which failures have an unknown server outcome.

    Only communication failures qualify; deterministic statement errors
    must propagate untouched.
    """

    def test_sqlalchemy_wrappers(self):
        self.assertTrue(pmod._is_connection_failure(
            OperationalError(None, {}, NetworkError('lost'))
        ))
        self.assertTrue(pmod._is_connection_failure(
            InterfaceError(None, {}, RuntimeError('closed'))
        ))

    def test_dialect_and_driver_errors(self):
        self.assertTrue(pmod._is_connection_failure(
            DBAPIError(None, {}, NetworkError('lost'))
        ))
        self.assertTrue(pmod._is_connection_failure(
            DBAPIError(None, {}, SocketTimeoutError('slow'))
        ))
        # The native connector wraps driver errors in DatabaseException;
        # sometimes nothing else wraps that one.
        self.assertTrue(pmod._is_connection_failure(
            DatabaseException(NetworkError('lost'))
        ))

    def test_http_layer_errors(self):
        from clickhouse_sqlalchemy.drivers.http.exceptions import (
            HTTPException
        )
        import requests

        self.assertTrue(pmod._is_connection_failure(
            DBAPIError(None, {}, HTTPException('reset'))
        ))
        self.assertTrue(pmod._is_connection_failure(
            DBAPIError(None, {}, requests.ConnectionError('refused'))
        ))
        self.assertTrue(pmod._is_connection_failure(
            DBAPIError(None, {}, requests.Timeout('slow'))
        ))

    def test_raw_builtin_connection_errors(self):
        # Observed when the server dies mid-query and the driver does not
        # wrap the socket error itself.
        self.assertTrue(pmod._is_connection_failure(
            ConnectionResetError(104, 'Connection reset by peer')
        ))
        self.assertTrue(pmod._is_connection_failure(
            BrokenPipeError(32, 'Broken pipe')
        ))
        self.assertTrue(pmod._is_connection_failure(TimeoutError('slow')))

    def test_deterministic_errors_are_not_connection_failures(self):
        self.assertFalse(pmod._is_connection_failure(
            DBAPIError(None, {}, ServerException('syntax error'))
        ))
        self.assertFalse(pmod._is_connection_failure(
            ProgrammingError(None, {}, Exception('bad statement'))
        ))
        self.assertFalse(pmod._is_connection_failure(
            DatabaseException(ServerException('no such table'))
        ))
        self.assertFalse(pmod._is_connection_failure(ValueError('bug')))
