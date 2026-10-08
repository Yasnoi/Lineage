"""Current-prefix, two-lane, original-record and recovery contracts."""
import asyncio
from benchmarks.checks.support.compaction import CHECK_DIR, run_case
from benchmarks.checks.support.events import check_group, group_id
from benchmarks.tests import test_prefix_compaction as prefix

async def check_prefix_independent_long_run_recovers_current_prefix_and_protects_three_groups():
    await run_case(prefix.PrefixCompactionTests, "test_independent_long_run_recovers_current_prefix_and_protects_three_groups")

async def check_prefix_session_first_long_run_originals_and_r_inherit_with_old_turn():
    await run_case(prefix.PrefixCompactionTests, "test_session_first_long_run_originals_and_r_inherit_with_old_turn")

async def check_prefix_original_call_ids_remain_reserved_after_prefix_compaction():
    await run_case(prefix.PrefixCompactionTests, "test_original_call_ids_remain_reserved_after_prefix_compaction")

async def check_prefix_summary_failure_keeps_active_originals_and_does_not_replay_tools():
    await run_case(prefix.PrefixCompactionTests, "test_summary_failure_keeps_active_originals_and_does_not_replay_tools")

async def check_prefix_history_and_current_prefix_generate_h_then_r_and_commit_together():
    await run_case(prefix.PrefixCompactionTests, "test_history_and_current_prefix_generate_h_then_r_and_commit_together")

async def check_prefix_h_success_r_failure_keeps_both_original_views():
    await run_case(prefix.PrefixCompactionTests, "test_h_success_r_failure_keeps_both_original_views")

async def check_prefix_failed_first_model_call_history_keeps_error_outcome_when_compacted():
    await run_case(prefix.PrefixCompactionTests, "test_failed_first_model_call_history_keeps_error_outcome_when_compacted")

async def check_prefix_current_protection_never_preserves_historical_turns():
    await run_case(prefix.PrefixCompactionTests, "test_current_protection_never_preserves_historical_turns")

async def check_prefix_model_searches_and_reads_detail_missing_from_r_without_tool_replay():
    await run_case(prefix.PrefixCompactionTests, "test_model_searches_and_reads_detail_missing_from_r_without_tool_replay")

async def check_prefix_oversized_valid_candidate_is_shortened_once_without_original_bodies():
    await run_case(prefix.PrefixCompactionTests, "test_oversized_valid_candidate_is_shortened_once_without_original_bodies")

async def check_prefix_empty_summary_stops_after_one_call_and_keeps_original_view():
    await run_case(prefix.PrefixCompactionTests, "test_empty_summary_stops_after_one_call_and_keeps_original_view")

async def check_prefix_checkpoint_failures_keep_old_activity_and_actual_tool_and_summary_facts():
    await run_case(prefix.PrefixCompactionTests, "test_checkpoint_failures_keep_old_activity_and_actual_tool_and_summary_facts")

async def check_prefix_two_complete_attempts_can_call_both_lanes_but_never_a_fifth_summary():
    await run_case(prefix.PrefixCompactionTests, "test_two_complete_attempts_can_call_both_lanes_but_never_a_fifth_summary")

async def check_prefix_second_oversized_candidate_stops_without_reinitializing_attempts():
    await run_case(prefix.PrefixCompactionTests, "test_second_oversized_candidate_stops_without_reinitializing_attempts")

async def check_prefix_source_version_change_preserves_latest_confirmed_view():
    await run_case(prefix.PrefixCompactionTests, "test_source_version_change_preserves_latest_confirmed_view")

async def check_prefix_protected_context_too_large_stops_without_reducing_three_groups():
    await run_case(prefix.PrefixCompactionTests, "test_protected_context_too_large_stops_without_reducing_three_groups")

async def check_prefix_summary_request_preflight_refuses_without_sending_or_recursive_recovery():
    await run_case(prefix.PrefixCompactionTests, "test_summary_request_preflight_refuses_without_sending_or_recursive_recovery")

async def check_prefix_multiple_tool_calls_and_legal_failures_remain_in_whole_groups():
    await run_case(prefix.PrefixCompactionTests, "test_multiple_tool_calls_and_legal_failures_remain_in_whole_groups")

async def check_prefix_unclosed_multi_tool_group_is_excluded_from_confirmed_prefix_selection():
    await run_case(prefix.PrefixCompactionTests, "test_unclosed_multi_tool_group_is_excluded_from_confirmed_prefix_selection")

async def check_prefix_previous_h_is_reused_without_call_or_target_allocation():
    await run_case(prefix.PrefixCompactionTests, "test_previous_h_is_reused_without_call_or_target_allocation")

async def check_prefix_actual_summary_request_larger_than_window_is_not_sent():
    await run_case(prefix.PrefixCompactionTests, "test_actual_summary_request_larger_than_window_is_not_sent")
from benchmarks.tests import test_compaction_boundaries as boundaries

async def check_boundaries_checked_requests_equal_actual_sdk_requests_including_final_call_options():
    await run_case(boundaries.CompactionBoundaryTests, "test_checked_requests_equal_actual_sdk_requests_including_final_call_options")

async def check_boundaries_allowed_candidate_below_old_headroom_commits_on_first_attempt():
    await run_case(boundaries.CompactionBoundaryTests, "test_allowed_candidate_below_old_headroom_commits_on_first_attempt")

async def check_boundaries_private_final_save_failure_cannot_report_completed():
    await run_case(boundaries.CompactionBoundaryTests, "test_private_final_save_failure_cannot_report_completed")

async def check_boundaries_activation_confirmation_failure_keeps_old_activity_and_paid_facts():
    await run_case(boundaries.CompactionBoundaryTests, "test_activation_confirmation_failure_keeps_old_activity_and_paid_facts")

async def check_boundaries_summary_logging_boundaries_do_not_send_or_enable_unsaved_work():
    await run_case(boundaries.CompactionBoundaryTests, "test_summary_logging_boundaries_do_not_send_or_enable_unsaved_work")

async def check_boundaries_short_run_has_no_summary_or_history_tools_executed():
    await run_case(boundaries.CompactionBoundaryTests, "test_short_run_has_no_summary_or_history_tools_executed")
from benchmarks.tests import test_history as history

async def check_history_records_survive_active_input_replacement_and_fixed_snapshot():
    await run_case(history.HistoryTests, "test_records_survive_active_input_replacement_and_fixed_snapshot")

async def check_history_original_record_creation_preserves_invalid_input_for_admission():
    await run_case(history.HistoryTests, "test_original_record_creation_preserves_invalid_input_for_admission")

async def check_history_native_saved_non_json_opaque_values_do_not_poison_later_records():
    await run_case(history.HistoryTests, "test_native_saved_non_json_opaque_values_do_not_poison_later_records")

async def check_history_missing_sequences_and_conflicting_immutable_identity_fail():
    await run_case(history.HistoryTests, "test_missing_sequences_and_conflicting_immutable_identity_fail")

async def check_history_unconfirmed_checkpoints_and_released_cache_are_unreadable():
    await run_case(history.HistoryTests, "test_unconfirmed_checkpoints_and_released_cache_are_unreadable")

async def check_history_scope_is_immutable_and_contains_unreferenced_originals():
    await run_case(history.HistoryTests, "test_scope_is_immutable_and_contains_unreferenced_originals")

async def check_history_read_unicode_escape_pagination_matches_fixed_projection():
    await run_case(history.HistoryTests, "test_read_unicode_escape_pagination_matches_fixed_projection")

async def check_history_search_pages_have_no_omissions_and_cursors_are_bound():
    await run_case(history.HistoryTests, "test_search_pages_have_no_omissions_and_cursors_are_bound")

async def check_history_cross_session_reference_is_rejected():
    await run_case(history.HistoryTests, "test_cross_session_reference_is_rejected")

async def check_history_native_history_recall_has_no_artifact_or_file_budget():
    await run_case(history.HistoryTests, "test_native_history_recall_has_no_artifact_or_file_budget")

async def check_history_opaque_projection_omits_encrypted_and_unknown_content():
    await run_case(history.HistoryTests, "test_opaque_projection_omits_encrypted_and_unknown_content")

async def check_history_tool_facts_distinguish_execution_and_observation_failure():
    await run_case(history.HistoryTests, "test_tool_facts_distinguish_execution_and_observation_failure")

async def check_history_interruption_keeps_actual_facts_without_fabricating_results():
    await run_case(history.HistoryTests, "test_interruption_keeps_actual_facts_without_fabricating_results")
from benchmarks.tests import test_compact_settings as settings

async def check_settings_defaults_and_limits():
    await run_case(settings.CompactSettingsTests, "test_defaults_and_limits")

async def check_settings_old_turn_setting_is_rejected_in_constructor_and_environment():
    await run_case(settings.CompactSettingsTests, "test_old_turn_setting_is_rejected_in_constructor_and_environment")

async def check_settings_new_environment_values_are_loaded_without_old_field():
    await run_case(settings.CompactSettingsTests, "test_new_environment_values_are_loaded_without_old_field")
from benchmarks.tests import test_summary_contracts as summary

async def check_summary_lane_caps_are_independent_of_business_output_allowance():
    await run_case(summary.SummaryContractsTests, "test_lane_caps_are_independent_of_business_output_allowance")

async def check_summary_invalid_lane_or_attempt_never_prepares_a_request():
    await run_case(summary.SummaryContractsTests, "test_invalid_lane_or_attempt_never_prepares_a_request")

async def check_summary_plain_text_is_preserved_without_schema_or_reference_parsing():
    await run_case(summary.SummaryContractsTests, "test_plain_text_is_preserved_without_schema_or_reference_parsing")

async def check_summary_invalid_completion_refusal_tool_request_and_empty_text_are_rejected():
    await run_case(summary.SummaryContractsTests, "test_invalid_completion_refusal_tool_request_and_empty_text_are_rejected")

async def check_summary_summary_budget_uses_lane_cap_without_changing_business_settings():
    await run_case(summary.SummaryContractsTests, "test_summary_budget_uses_lane_cap_without_changing_business_settings")

async def check_summary_complete_multi_tool_group_including_failure_is_allowed():
    await run_case(summary.GroupProtocolTests, "test_complete_multi_tool_group_including_failure_is_allowed")

async def check_summary_known_encrypted_reasoning_is_whole_group_only():
    await run_case(summary.GroupProtocolTests, "test_known_encrypted_reasoning_is_whole_group_only")

async def check_summary_unrecognized_nontext_authority_and_incomplete_items_are_barriers():
    await run_case(summary.GroupProtocolTests, "test_unrecognized_nontext_authority_and_incomplete_items_are_barriers")

from benchmarks.tests import test_summary_input as text_input

async def check_text_interleaved_calls_reversed_results_and_reused_ids_across_runs():
    await run_case(text_input.SummaryInputTests, "test_interleaved_calls_reversed_results_and_reused_ids_across_runs")

async def check_text_readable_reasoning_summary_and_opaque_placeholder_keep_their_positions():
    await run_case(text_input.SummaryInputTests, "test_readable_reasoning_summary_and_opaque_placeholder_keep_their_positions")

async def check_text_tags_and_long_tool_text_are_data_without_truncation_or_nested_json_encoding():
    await run_case(text_input.SummaryInputTests, "test_tags_and_long_tool_text_are_data_without_truncation_or_nested_json_encoding")

async def check_text_scope_background_and_previous_summary_use_trusted_identity_only():
    await run_case(text_input.SummaryInputTests, "test_scope_background_and_previous_summary_use_trusted_identity_only")

async def check_text_unknown_dependencies_unclosed_pairs_and_system_instructions_still_fail():
    await run_case(text_input.SummaryInputTests, "test_unknown_dependencies_unclosed_pairs_and_system_instructions_still_fail")

async def main():
    checks = (
        check_prefix_independent_long_run_recovers_current_prefix_and_protects_three_groups,
        check_prefix_session_first_long_run_originals_and_r_inherit_with_old_turn,
        check_prefix_original_call_ids_remain_reserved_after_prefix_compaction,
        check_prefix_summary_failure_keeps_active_originals_and_does_not_replay_tools,
        check_prefix_history_and_current_prefix_generate_h_then_r_and_commit_together,
        check_prefix_h_success_r_failure_keeps_both_original_views,
        check_prefix_failed_first_model_call_history_keeps_error_outcome_when_compacted,
        check_prefix_current_protection_never_preserves_historical_turns,
        check_prefix_model_searches_and_reads_detail_missing_from_r_without_tool_replay,
        check_prefix_oversized_valid_candidate_is_shortened_once_without_original_bodies,
        check_prefix_empty_summary_stops_after_one_call_and_keeps_original_view,
        check_prefix_checkpoint_failures_keep_old_activity_and_actual_tool_and_summary_facts,
        check_prefix_two_complete_attempts_can_call_both_lanes_but_never_a_fifth_summary,
        check_prefix_second_oversized_candidate_stops_without_reinitializing_attempts,
        check_prefix_source_version_change_preserves_latest_confirmed_view,
        check_prefix_protected_context_too_large_stops_without_reducing_three_groups,
        check_prefix_summary_request_preflight_refuses_without_sending_or_recursive_recovery,
        check_prefix_multiple_tool_calls_and_legal_failures_remain_in_whole_groups,
        check_prefix_unclosed_multi_tool_group_is_excluded_from_confirmed_prefix_selection,
        check_prefix_previous_h_is_reused_without_call_or_target_allocation,
        check_prefix_actual_summary_request_larger_than_window_is_not_sent,
        check_boundaries_checked_requests_equal_actual_sdk_requests_including_final_call_options,
        check_boundaries_allowed_candidate_below_old_headroom_commits_on_first_attempt,
        check_boundaries_private_final_save_failure_cannot_report_completed,
        check_boundaries_activation_confirmation_failure_keeps_old_activity_and_paid_facts,
        check_boundaries_summary_logging_boundaries_do_not_send_or_enable_unsaved_work,
        check_boundaries_short_run_has_no_summary_or_history_tools_executed,
        check_history_records_survive_active_input_replacement_and_fixed_snapshot,
        check_history_original_record_creation_preserves_invalid_input_for_admission,
        check_history_native_saved_non_json_opaque_values_do_not_poison_later_records,
        check_history_missing_sequences_and_conflicting_immutable_identity_fail,
        check_history_unconfirmed_checkpoints_and_released_cache_are_unreadable,
        check_history_scope_is_immutable_and_contains_unreferenced_originals,
        check_history_read_unicode_escape_pagination_matches_fixed_projection,
        check_history_search_pages_have_no_omissions_and_cursors_are_bound,
        check_history_cross_session_reference_is_rejected,
        check_history_native_history_recall_has_no_artifact_or_file_budget,
        check_history_opaque_projection_omits_encrypted_and_unknown_content,
        check_history_tool_facts_distinguish_execution_and_observation_failure,
        check_history_interruption_keeps_actual_facts_without_fabricating_results,
        check_settings_defaults_and_limits,
        check_settings_old_turn_setting_is_rejected_in_constructor_and_environment,
        check_settings_new_environment_values_are_loaded_without_old_field,
        check_summary_lane_caps_are_independent_of_business_output_allowance,
        check_summary_invalid_lane_or_attempt_never_prepares_a_request,
        check_summary_plain_text_is_preserved_without_schema_or_reference_parsing,
        check_summary_invalid_completion_refusal_tool_request_and_empty_text_are_rejected,
        check_summary_summary_budget_uses_lane_cap_without_changing_business_settings,
        check_summary_complete_multi_tool_group_including_failure_is_allowed,
        check_summary_known_encrypted_reasoning_is_whole_group_only,
        check_summary_unrecognized_nontext_authority_and_incomplete_items_are_barriers,
        check_text_interleaved_calls_reversed_results_and_reused_ids_across_runs,
        check_text_readable_reasoning_summary_and_opaque_placeholder_keep_their_positions,
        check_text_tags_and_long_tool_text_are_data_without_truncation_or_nested_json_encoding,
        check_text_scope_background_and_previous_summary_use_trusted_identity_only,
        check_text_unknown_dependencies_unclosed_pairs_and_system_instructions_still_fail,
    )
    for check in checks:
        with check_group(group_id(check)):
            await check()
        print(f"PASS {check.__name__}")
    print(f"PASS {len(checks)} Compact groups; logs: {CHECK_DIR}")

def execute():
    asyncio.run(main())
