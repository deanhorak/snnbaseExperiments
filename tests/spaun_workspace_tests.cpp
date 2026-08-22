#include <snnbase_experiments/spaun_workspace.hpp>

#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace {

using snnbase_experiments::spaun::workspace::Pathway;
using snnbase_experiments::spaun::workspace::Workspace;
using snnbase_experiments::spaun::workspace::digit_event;
using snnbase_experiments::spaun::workspace::group_boundary_event;
using snnbase_experiments::spaun::workspace::query_event;
using snnbase_experiments::spaun::workspace::sequence_restart_event;
using snnbase_experiments::spaun::workspace::task_event;
using snnbase_experiments::spaun::workspace::vocabulary;

void fixed_vocabulary_contains_every_spaun_control_family() {
  using namespace snnbase_experiments::spaun::workspace;
  assert(vocabulary().dimension() == semantic_dimensions);
  assert(vocabulary().size() == counting_number_count + spaun_task_count +
                                      query_kind_count + position_count);
  for (std::size_t digit = 0; digit < decimal_digit_count; ++digit) {
    assert(vocabulary().contains(digit_name(digit)));
  }
  for (std::size_t task = 0; task < spaun_task_count; ++task) {
    assert(vocabulary().contains(task_name(task)));
  }
  assert(vocabulary().contains(query_name('P')));
  assert(vocabulary().contains(query_name('K')));
  for (std::size_t position = 1; position <= position_count; ++position) {
    assert(vocabulary().contains(position_name(position)));
  }
}

void delayed_successor_chain_performs_neural_counting() {
  using snnbase_experiments::spaun::workspace::Config;
  const auto run_count = [](const std::size_t start,
                            const std::size_t increments) {
    Workspace memory(Config{.counting_delay_ticks = 3, .seed = 77});
    memory.accept(digit_event(start));
    memory.accept(group_boundary_event());
    memory.accept(digit_event(increments));
    assert(memory.begin_counting_from_memory(0, 0));
    memory.advance(memory.counting_delay_ticks() * increments);
    return memory.route_counting_result(increments);
  };

  assert(run_count(3, 5) == 8);
  assert(run_count(7, 5) == 12);

  Workspace lesioned(Config{.counting_delay_ticks = 2});
  lesioned.accept(digit_event(4));
  lesioned.set_pathway_enabled(Pathway::working_memory_to_transformation, false);
  assert(!lesioned.begin_counting_from_memory(0, 0));
  lesioned.advance(4);
  assert(!lesioned.route_counting_result(2).has_value());
}

void nine_groups_of_nine_digits_are_independently_recalled() {
  Workspace memory;
  assert(memory.maximum_groups() >= 9);
  assert(memory.maximum_list_length() >= 9);

  for (std::size_t group = 0; group < 9; ++group) {
    for (std::size_t item = 0; item < 9; ++item) {
      memory.accept(digit_event((group * 3 + item) % 10));
    }
    if (group + 1 < 9) {
      memory.accept(group_boundary_event());
    }
  }
  memory.advance(12);

  for (std::size_t group = 0; group < 9; ++group) {
    assert(memory.stored_item_count(group) == 9);
    const auto recalled = memory.recall_group(group);
    assert(recalled.size() == 9);
    for (std::size_t item = 0; item < 9; ++item) {
      assert(recalled[item] == static_cast<int>((group * 3 + item) % 10));
    }
  }
  assert(memory.population_count() >= 89);
  assert(memory.neuron_count() > 0);
  assert(memory.projection_count() > 0);
}

void event_driven_restart_overwrites_slots_and_reset_clears_everything() {
  Workspace memory;
  memory.accept(digit_event(1));
  memory.accept(digit_event(2));
  assert(memory.recall_digit(0, 0) == 1);
  assert(memory.recall_digit(0, 1) == 2);

  memory.accept(sequence_restart_event());
  memory.accept(digit_event(8));
  assert(memory.stored_item_count(0) == 1);
  assert(memory.recall_digit(0, 0) == 8);
  assert(!memory.recall_digit(0, 1).has_value());

  memory.accept(task_event(2));
  memory.accept(task_event(7));
  memory.accept(query_event('P'));
  memory.accept(query_event('K'));
  memory.advance(10);
  assert(memory.task_state() == 7);
  assert(memory.query_state() == 'K');

  memory.reset();
  assert(memory.time() == 0);
  assert(memory.current_group() == 0);
  assert(memory.current_item() == 0);
  assert(memory.stored_item_count(0) == 0);
  assert(!memory.recall_digit(0, 0).has_value());
  assert(!memory.task_state().has_value());
  assert(!memory.query_state().has_value());
}

void recurrence_is_required_for_persistence_and_task_state() {
  Workspace memory;
  memory.accept(task_event(3));
  memory.accept(digit_event(4));
  memory.accept(digit_event(6));
  memory.advance(20);
  assert(memory.task_state() == 3);
  assert(memory.recall_digit(0, 0) == 4);
  assert(memory.recall_digit(0, 1) == 6);

  memory.set_pathway_enabled(Pathway::working_memory_recurrence, false);
  assert(!memory.pathway_enabled(Pathway::working_memory_recurrence));
  assert(!memory.task_state().has_value());
  assert(!memory.recall_digit(0, 0).has_value());
  assert(!memory.recall_digit(0, 1).has_value());
  assert(!memory.route_recalled_digit(0, 0).has_value());

  // Re-enabling a severed recurrent pathway cannot reconstruct lost state.
  memory.set_pathway_enabled(Pathway::working_memory_recurrence, true);
  memory.advance(4);
  assert(!memory.recall_digit(0, 0).has_value());
}

void motor_output_is_decoded_only_after_the_full_spiking_route() {
  Workspace memory;
  for (std::size_t digit = 0; digit < 10; ++digit) {
    const auto& planned = vocabulary().get(
        snnbase_experiments::spaun::workspace::digit_name(digit));
    assert(memory.route_planned_digit(planned) == static_cast<int>(digit));
  }

  memory.accept(digit_event(7));
  memory.accept(digit_event(0));
  assert(memory.route_recalled_digit(0, 0) == 7);
  assert(memory.route_recalled_digit(0, 1) == 0);
}

void every_output_pathway_has_a_causal_lesion_effect() {
  const auto& digit = vocabulary().get("D5");

  {
    Workspace memory;
    memory.set_pathway_enabled(Pathway::encoding_to_working_memory, false);
    assert(!memory.route_planned_digit(digit).has_value());
  }
  {
    Workspace memory;
    memory.set_pathway_enabled(Pathway::working_memory_recurrence, false);
    assert(!memory.route_planned_digit(digit).has_value());
  }
  {
    Workspace memory;
    memory.set_pathway_enabled(
        Pathway::working_memory_to_transformation, false);
    assert(!memory.route_planned_digit(digit).has_value());
  }
  {
    Workspace memory;
    memory.set_pathway_enabled(
        Pathway::transformation_action_to_decoding, false);
    assert(!memory.route_planned_digit(digit).has_value());
  }
  {
    Workspace memory;
    memory.set_pathway_enabled(Pathway::decoding_to_motor, false);
    assert(!memory.route_planned_digit(digit).has_value());
  }
}

void identical_event_streams_are_deterministic() {
  Workspace first;
  Workspace second;
  for (const auto digit : {9U, 1U, 4U, 7U}) {
    first.accept(digit_event(digit));
    second.accept(digit_event(digit));
  }
  first.accept(task_event(6));
  second.accept(task_event(6));
  first.advance(7);
  second.advance(7);

  assert(first.recall_group(0) == second.recall_group(0));
  assert(first.task_state() == second.task_state());
  assert(first.route_recalled_digit(0, 2) ==
         second.route_recalled_digit(0, 2));
  assert(first.time() == second.time());
  assert(first.neuron_count() == second.neuron_count());
}

void malformed_capacity_and_events_are_rejected() {
  using namespace snnbase_experiments::spaun::workspace;
  bool threw = false;
  try {
    static_cast<void>(Workspace(Config{.maximum_groups = 8}));
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  assert(threw);

  Workspace memory;
  threw = false;
  try {
    memory.accept(EncodedTokenEvent{TokenKind::digit, std::nullopt});
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  assert(threw);
}

}  // namespace

int main() {
  fixed_vocabulary_contains_every_spaun_control_family();
  nine_groups_of_nine_digits_are_independently_recalled();
  event_driven_restart_overwrites_slots_and_reset_clears_everything();
  recurrence_is_required_for_persistence_and_task_state();
  motor_output_is_decoded_only_after_the_full_spiking_route();
  delayed_successor_chain_performs_neural_counting();
  every_output_pathway_has_a_causal_lesion_effect();
  identical_event_streams_are_deterministic();
  malformed_capacity_and_events_are_rejected();
}
