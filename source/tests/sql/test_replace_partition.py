from clickhouse_sqlalchemy.sql.ddl import ReplacePartition
from tests.testcase import CompilationTestCase


class ReplacePartitionCompilationTestCase(CompilationTestCase):
    def test_basic(self):
        statement = ReplacePartition('target', 'staging', '202601')
        self.assertEqual(
            self.compile(statement),
            'ALTER TABLE "target" REPLACE PARTITION \'202601\' '
            'FROM "staging"'
        )

    def test_with_schema(self):
        statement = ReplacePartition(
            'target', 'staging', '202601',
            target_schema='db', source_schema='db'
        )
        self.assertEqual(
            self.compile(statement),
            'ALTER TABLE db."target" REPLACE PARTITION \'202601\' '
            'FROM db."staging"'
        )

    def test_integer_partition_id_is_string_literal(self):
        statement = ReplacePartition('target', 'staging', 5)
        self.assertEqual(
            self.compile(statement),
            'ALTER TABLE "target" REPLACE PARTITION \'5\' FROM "staging"'
        )
        self.assertEqual(statement.partition_id, '5')

    def test_quoting_and_escaping(self):
        statement = ReplacePartition('ta`ble', 'st`aging', "a'b")
        self.assertEqual(
            self.compile(statement),
            'ALTER TABLE "ta`ble" REPLACE PARTITION \'a\'\'b\' '
            'FROM "st`aging"'
        )
