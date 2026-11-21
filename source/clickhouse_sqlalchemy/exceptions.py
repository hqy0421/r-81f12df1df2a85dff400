

class DatabaseException(Exception):
    def __init__(self, orig):
        self.orig = orig
        super(DatabaseException, self).__init__(orig)

    def __str__(self):
        return 'Orig exception: {}'.format(self.orig)


class PartitionPublishError(Exception):
    """Base class for partition publication errors."""


class PartitionPrecheckError(PartitionPublishError):
    """A read-only publication precheck has failed.

    No statement that modifies data has been executed.
    """

    def __init__(self, failures):
        if isinstance(failures, str):
            failures = [failures]
        self.failures = list(failures)
        super(PartitionPrecheckError, self).__init__(
            '; '.join(self.failures)
        )


class PartitionPublishUnknownError(PartitionPublishError):
    """The publication statement outcome cannot be confirmed.

    Raised when the connection is lost (or another communication error
    occurs) while the single ``REPLACE PARTITION`` statement is in flight.
    The server may or may not have applied it. The publication plan and the
    staging table are kept untouched so that the very same plan can be
    retried safely: ``REPLACE PARTITION`` replaces the partition atomically
    and does not accumulate rows when re-executed.
    """

    def __init__(self, message, original_error, plan):
        self.original_error = original_error
        self.plan = plan
        super(PartitionPublishUnknownError, self).__init__(message)
