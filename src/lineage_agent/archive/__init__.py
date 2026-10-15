"""Conversation business records, independent of graph checkpoint history.

ArchiveStore binds one trusted conversation directory; create=False opens an
existing versioned database without discovering or authorizing body files.
Consumers must retain a confirmed view_id separately from this binding:

* summary(id, view_id) and scope(ref, view_id) read immutable provenance.
* RecordStore(archive=archive, view_id=view_id) rebuilds History and its cursors.
* ArtifactStore.restore_session(alias, archive=archive, view_id=view_id) restores
  only the registered Artifact references accepted by that view.
* read_archived_record(archive, run_id, view_id) reads sealed run details.

Runtime writers append events and register bodies before prepare_view. State
save precedes confirm_view; both must succeed before advancing live access.
register_conversation records the non-secret recovery descriptor; begin_run
records an attempt before business execution. After State readback,
publish_snapshot atomically seals the staged Run, publishes its immutable
snapshot and advances current_snapshot_id. Resource views never imply a new
recovery point. read_snapshot validates the selected point and published_runs
returns only its recognized ancestors. accept_recovery records an explicit
choice to leave unfinished attempts outside that ancestry.
stage_run/seal_run remain lower-level business-detail APIs.
The raw events/read_run APIs are trusted archive inspection interfaces, not
model tools or permission to adopt unconfirmed facts. No method resumes a graph.
"""

from lineage_agent.archive.store import ArchiveError, ArchiveStore, FORMAT_VERSION, RECOVERY_VERSION

__all__ = ["ArchiveError", "ArchiveStore", "FORMAT_VERSION", "RECOVERY_VERSION"]
