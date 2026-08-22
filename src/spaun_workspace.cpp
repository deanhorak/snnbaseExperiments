#include <snnbase_experiments/spaun_workspace.hpp>

#include <snnbase/semantic_population.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace snnbase_experiments::spaun::workspace {
namespace {

using snnbase::semantic_pointer::Pointer;
using snnbase::semantic_pointer::Vocabulary;
using snnbase::semantic_population::ConnectionConfig;
using snnbase::semantic_population::SemanticNetwork;
using snnbase::semantic_population::SemanticPopulationConfig;
using snnbase::semantic_population::SemanticPopulationId;
using snnbase::semantic_population::SemanticProjectionId;

constexpr std::array<std::string_view, decimal_digit_count> digit_names{
    "D0", "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9"};
constexpr std::array<std::string_view, counting_number_count> number_names{
    "D0",  "D1",  "D2",  "D3",  "D4",  "D5",  "D6",
    "D7",  "D8",  "D9",  "N10", "N11", "N12", "N13",
    "N14", "N15", "N16", "N17", "N18"};
constexpr std::array<std::string_view, spaun_task_count> task_names{
    "A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7"};
constexpr std::array<std::string_view, query_kind_count> query_names{"P", "K"};
constexpr std::array<std::string_view, position_count> position_names{
    "POS1", "POS2", "POS3", "POS4", "POS5",
    "POS6", "POS7", "POS8", "POS9"};

constexpr std::size_t pathway_count = 5;

std::size_t pathway_index(const Pathway pathway) {
  const auto index = static_cast<std::size_t>(pathway);
  if (index >= pathway_count) {
    throw std::invalid_argument("unknown Spaun workspace pathway");
  }
  return index;
}

void require_finite(const double value, const char* description) {
  if (!std::isfinite(value)) {
    throw std::invalid_argument(std::string(description) + " must be finite");
  }
}

std::vector<double> permutation_transform(const std::size_t dimensions,
                                          const int shift) {
  std::vector<double> transform(dimensions * dimensions, 0.0);
  const auto signed_dimensions = static_cast<long long>(dimensions);
  const auto normalized =
      ((static_cast<long long>(shift) % signed_dimensions) +
       signed_dimensions) %
      signed_dimensions;
  for (std::size_t source = 0; source < dimensions; ++source) {
    const auto target = static_cast<std::size_t>(
        (static_cast<long long>(source) + normalized) % signed_dimensions);
    transform[target * dimensions + source] = 1.0;
  }
  return transform;
}

std::vector<double> counting_successor_transform() {
  std::vector<Pointer> inputs;
  std::vector<Pointer> outputs;
  inputs.reserve(number_names.size() - 1U);
  outputs.reserve(number_names.size() - 1U);
  for (std::size_t number = 0; number + 1U < number_names.size(); ++number) {
    inputs.push_back(vocabulary().get(number_names[number]));
    outputs.push_back(vocabulary().get(number_names[number + 1U]));
  }
  return snnbase::semantic_pointer::fit_linear_transform(inputs, outputs);
}

template <std::size_t Size>
std::string_view checked_name(const std::array<std::string_view, Size>& names,
                              const std::size_t index,
                              const char* description) {
  if (index >= names.size()) {
    throw std::out_of_range(std::string(description) + " is out of range");
  }
  return names[index];
}

}  // namespace

const Vocabulary& vocabulary() {
  static const Vocabulary fixed = [] {
    Vocabulary result(semantic_dimensions, 0x535041554eULL);
    for (const auto name : digit_names) {
      static_cast<void>(result.add_random(std::string(name)));
    }
    for (std::size_t number = decimal_digit_count;
         number < number_names.size(); ++number) {
      static_cast<void>(result.add_random(std::string(number_names[number])));
    }
    for (const auto name : task_names) {
      static_cast<void>(result.add_random(std::string(name)));
    }
    for (const auto name : query_names) {
      static_cast<void>(result.add_random(std::string(name)));
    }
    for (const auto name : position_names) {
      static_cast<void>(result.add_random(std::string(name)));
    }
    return result;
  }();
  return fixed;
}

std::string_view digit_name(const std::size_t digit) {
  return checked_name(digit_names, digit, "digit");
}

std::string_view task_name(const std::size_t task) {
  return checked_name(task_names, task, "task");
}

std::string_view query_name(const char query) {
  if (query == 'P' || query == 'p') {
    return query_names[0];
  }
  if (query == 'K' || query == 'k') {
    return query_names[1];
  }
  throw std::out_of_range("query kind must be P or K");
}

std::string_view position_name(const std::size_t one_based_position) {
  if (one_based_position == 0) {
    throw std::out_of_range("position must be in the range 1--9");
  }
  return checked_name(position_names, one_based_position - 1, "position");
}

EncodedTokenEvent digit_event(const std::size_t digit) {
  return {TokenKind::digit, vocabulary().get(digit_name(digit))};
}

EncodedTokenEvent task_event(const std::size_t task) {
  return {TokenKind::task, vocabulary().get(task_name(task))};
}

EncodedTokenEvent query_event(const char query) {
  return {TokenKind::query, vocabulary().get(query_name(query))};
}

EncodedTokenEvent group_boundary_event() {
  return {TokenKind::group_boundary, std::nullopt};
}

EncodedTokenEvent sequence_restart_event() {
  return {TokenKind::sequence_restart, std::nullopt};
}

struct Workspace::Impl {
  explicit Impl(Config workspace_config) : config(std::move(workspace_config)) {
    validate_config();
    pathway_state.fill(true);
    item_counts.assign(config.maximum_groups, 0);

    encoding = add_population(false);
    task = add_population(false);
    query = add_population(false);
    plan = add_population(false);
    transformation = add_population(false);
    action = add_population(false);
    decoding = add_population(false);
    motor = add_population(false);

    const auto slot_count =
        config.maximum_groups * config.maximum_list_length;
    slots.reserve(slot_count);
    for (std::size_t index = 0; index < slot_count; ++index) {
      slots.push_back(add_population(true));
    }

    counting_stages.reserve(config.maximum_list_length + 1U);
    for (std::size_t stage = 0; stage <= config.maximum_list_length; ++stage) {
      counting_stages.push_back(add_population(false));
    }
    for (const auto stage : counting_stages) {
      const auto recurrent = network.connect_recurrent(
          stage, 1.0, connection(true, config.projection_weight_gain));
      counting_recurrence.push_back(recurrent);
      recurrence_projections.push_back(recurrent);
    }

    encoding_to_slots.reserve(slot_count);
    slot_recurrence.reserve(slot_count);
    slots_to_transformation.reserve(slot_count);
    slots_to_counting_start.reserve(slot_count);
    for (std::size_t index = 0; index < slots.size(); ++index) {
      const auto slot = slots[index];
      const auto encoding_path = network.connect_identity(
          encoding, slot, connection(false));
      encoding_to_slots.push_back(encoding_path);
      encoding_projections.push_back(encoding_path);

      const auto serial_position = index % config.maximum_list_length;
      const auto recurrent_gain =
          serial_position == 0 ? config.primacy_recurrent_weight_gain
                               : config.memory_recurrent_weight_gain;
      const auto recurrent = network.connect_recurrent(
          slot, 1.0, connection(true, recurrent_gain));
      slot_recurrence.push_back(recurrent);
      recurrence_projections.push_back(recurrent);

      const auto transform = permutation_transform(semantic_dimensions, 1);
      const auto output_path = network.connect_linear(
          slot, transformation, transform, connection(false));
      slots_to_transformation.push_back(output_path);
      wm_to_transformation_projections.push_back(output_path);

      const auto counting_start = network.connect_identity(
          slot, counting_stages.front(), connection(false));
      slots_to_counting_start.push_back(counting_start);
      wm_to_transformation_projections.push_back(counting_start);
    }


    const auto successor = counting_successor_transform();
    counting_successors.reserve(config.maximum_list_length);
    for (std::size_t stage = 0; stage + 1U < counting_stages.size(); ++stage) {
      counting_successors.push_back(network.connect_linear(
          counting_stages[stage], counting_stages[stage + 1U], successor,
          connection(true, config.projection_weight_gain,
                     config.counting_delay_ticks)));
    }
    counting_to_transformation.reserve(counting_stages.size());
    const auto counting_forward = permutation_transform(semantic_dimensions, 1);
    for (const auto stage : counting_stages) {
      const auto output_path = network.connect_linear(
          stage, transformation, counting_forward, connection(false));
      counting_to_transformation.push_back(output_path);
      wm_to_transformation_projections.push_back(output_path);
    }

    encoding_to_task =
        network.connect_identity(encoding, task, connection(false));
    encoding_to_query =
        network.connect_identity(encoding, query, connection(false));
    encoding_to_plan =
        network.connect_identity(encoding, plan, connection(false));
    encoding_projections.push_back(encoding_to_task);
    encoding_projections.push_back(encoding_to_query);
    encoding_projections.push_back(encoding_to_plan);

    task_recurrence =
        network.connect_recurrent(task, 1.0, connection(true));
    query_recurrence =
        network.connect_recurrent(query, 1.0, connection(true));
    plan_recurrence =
        network.connect_recurrent(plan, 1.0, connection(true));
    recurrence_projections.push_back(task_recurrence);
    recurrence_projections.push_back(query_recurrence);
    recurrence_projections.push_back(plan_recurrence);

    const auto forward = permutation_transform(semantic_dimensions, 1);
    plan_to_transformation = network.connect_linear(
        plan, transformation, forward, connection(false));
    wm_to_transformation_projections.push_back(plan_to_transformation);

    // Action selection applies the inverse semantic permutation.  These are
    // real signed linear transforms compiled into opponent-coded synapses.
    const auto inverse = permutation_transform(semantic_dimensions, -1);
    transformation_to_action = network.connect_linear(
        transformation, action, inverse, connection(true));
    action_to_decoding =
        network.connect_identity(action, decoding, connection(true));
    decoding_to_motor =
        network.connect_identity(decoding, motor, connection(true));
  }

  void validate_config() const {
    if (config.maximum_groups < position_count) {
      throw std::invalid_argument(
          "Spaun workspace must support at least nine groups");
    }
    if (config.maximum_list_length < position_count) {
      throw std::invalid_argument(
          "Spaun workspace must support lists of at least nine items");
    }
    if (config.maximum_groups >
        std::numeric_limits<std::size_t>::max() /
            config.maximum_list_length) {
      throw std::overflow_error("Spaun workspace slot count overflows");
    }
    if (config.neurons_per_sign == 0) {
      throw std::invalid_argument("neurons per sign must be positive");
    }
    require_finite(config.semantic_input_gain, "semantic input gain");
    if (!(config.semantic_input_gain > 0.0)) {
      throw std::invalid_argument("semantic input gain must be positive");
    }
    require_finite(config.projection_weight_gain,
                   "projection weight gain");
    if (!(config.projection_weight_gain > 1.0) ||
        config.projection_weight_gain > 4.0) {
      throw std::invalid_argument(
          "projection weight gain must be in the range (1, 4]");
    }
    require_finite(config.memory_noise_standard_deviation,
                   "memory noise standard deviation");
    if (config.memory_noise_standard_deviation < 0.0) {
      throw std::invalid_argument(
          "memory noise standard deviation cannot be negative");
    }
    for (const auto& [gain, description] :
         std::array<std::pair<double, const char*>, 3>{
             {{config.memory_recurrent_weight_gain,
               "memory recurrent weight gain"},
              {config.primacy_recurrent_weight_gain,
               "primacy recurrent weight gain"},
              {config.recency_recurrent_weight_gain,
               "recency recurrent weight gain"}}}) {
      require_finite(gain, description);
      if (!(gain > 1.0) || gain > 4.0) {
        throw std::invalid_argument(std::string(description) +
                                    " must be in the range (1, 4]");
      }
    }
    require_finite(config.cleanup_similarity_threshold,
                   "cleanup similarity threshold");
    if (config.cleanup_similarity_threshold < -1.0 ||
        config.cleanup_similarity_threshold > 1.0) {
      throw std::invalid_argument(
          "cleanup similarity threshold must be in [-1, 1]");
    }
    if (config.counting_delay_ticks == 0) {
      throw std::invalid_argument("counting delay must be positive");
    }
  }

  SemanticPopulationId add_population(const bool memory_population) {
    const auto seed = config.seed + population_seed_offset++;
    return network.add_population(SemanticPopulationConfig{
        .dimensions = semantic_dimensions,
        .neurons_per_sign = config.neurons_per_sign,
        .input_gain = config.semantic_input_gain,
        .input_baseline = 0.0,
        .decoding_gain = 1.0,
        .threshold = 1.0,
        .leak = 0.0,
        .bias = 0.0,
        .noise_standard_deviation =
            memory_population ? config.memory_noise_standard_deviation : 0.0,
        .activity_decay = 0.0,
        .refractory_period = 0,
        .seed = seed,
    });
  }

  ConnectionConfig connection(
      const bool enabled,
      const double weight_gain = std::numeric_limits<double>::quiet_NaN(),
      const snnbase::TimeStep delay = 1) const {
    return {.weight_gain = std::isnan(weight_gain)
                               ? config.projection_weight_gain
                               : weight_gain,
            .delay = delay,
            .synaptic_decay = 0.0,
            .plasticity = snnbase::population::PlasticityRule::fixed,
            .learning_rate = 0.0,
            .eligibility_decay = 0.0,
            .minimum_weight = -4.0,
            .maximum_weight = 4.0,
            .enabled = enabled};
  }

  std::size_t slot_index(const std::size_t group,
                         const std::size_t item) const {
    if (group >= config.maximum_groups) {
      throw std::out_of_range("working-memory group is out of range");
    }
    if (item >= config.maximum_list_length) {
      throw std::out_of_range("working-memory item is out of range");
    }
    return group * config.maximum_list_length + item;
  }

  void require_semantic(const EncodedTokenEvent& event) const {
    if (!event.semantic.has_value()) {
      throw std::invalid_argument("semantic token event has no encoded value");
    }
    if (event.semantic->dimension() != semantic_dimensions) {
      throw std::invalid_argument(
          "token event has the wrong semantic dimension");
    }
  }

  void mark_recency(const std::size_t group, const std::size_t item_count) {
    if (item_count == 0 || group >= config.maximum_groups) {
      return;
    }
    const auto index = slot_index(group, item_count - 1);
    const auto raw_projection =
        network.raw_projection_id(slot_recurrence[index]);
    const auto current = network.raw_network().projection_weights(raw_projection);
    std::vector<double> weights(current.begin(), current.end());
    const auto recurrent_weight = config.recency_recurrent_weight_gain /
                                  static_cast<double>(config.neurons_per_sign);
    for (auto& weight : weights) {
      if (weight > 0.0) {
        weight = recurrent_weight;
      }
    }
    network.raw_network().set_projection_weights(raw_projection, weights);
  }

  void write_population(const SemanticProjectionId encoding_projection,
                        const SemanticProjectionId recurrent_projection,
                        const Pointer& value) {
    // Clear precisely this slot before writing.  With zero membrane leak and
    // zero activity decay, one blank step removes the old neural code.
    network.set_projection_enabled(recurrent_projection, false);
    static_cast<void>(network.step());
    const auto recurrence_enabled = pathway_state[pathway_index(
        Pathway::working_memory_recurrence)];
    network.set_projection_enabled(recurrent_projection, recurrence_enabled);

    const auto encoding_enabled = pathway_state[pathway_index(
        Pathway::encoding_to_working_memory)];
    network.set_projection_enabled(encoding_projection, encoding_enabled);
    static_cast<void>(network.step(encoding, value));
    static_cast<void>(network.step());
    network.set_projection_enabled(encoding_projection, false);

    // A final step proves that the representation is recurrently sustained,
    // rather than exposing the transient encoding projection as recall.
    static_cast<void>(network.step());
  }

  template <std::size_t Size>
  std::optional<int> cleanup_index(
      const Pointer& represented,
      const std::array<std::string_view, Size>& names) const {
    if (!(represented.norm() > 1.0e-12)) {
      return std::nullopt;
    }
    double best_similarity = -2.0;
    std::size_t best_index = 0;
    for (std::size_t index = 0; index < names.size(); ++index) {
      const auto similarity = snnbase::semantic_pointer::cosine_similarity(
          represented, vocabulary().get(names[index]));
      if (similarity > best_similarity) {
        best_similarity = similarity;
        best_index = index;
      }
    }
    if (best_similarity < config.cleanup_similarity_threshold) {
      return std::nullopt;
    }
    return static_cast<int>(best_index);
  }

  std::optional<int> cleanup_digit(const SemanticPopulationId population) const {
    return cleanup_index(network.decode(population), digit_names);
  }

  std::optional<int> cleanup_number(const SemanticPopulationId population) const {
    return cleanup_index(network.decode(population), number_names);
  }

  Pointer route(const SemanticProjectionId selected_path) {
    const auto route_enabled = pathway_state[pathway_index(
        Pathway::working_memory_to_transformation)];
    network.set_projection_enabled(selected_path, route_enabled);

    // The selected gate consumes the current WM spike event, rather than
    // resampling a noisy recurrent source. t0: transformation, t1: action,
    // t2: decoding, t3: motor.
    if (route_enabled) {
      network.enqueue_current_spikes(selected_path);
    }
    static_cast<void>(network.step());  // transformation
    network.set_projection_enabled(selected_path, false);
    static_cast<void>(network.step());  // action
    static_cast<void>(network.step());  // decoding
    static_cast<void>(network.step());  // motor
    return network.decode(motor);
  }

  void set_all(const std::vector<SemanticProjectionId>& projections,
               const bool enabled) {
    for (const auto projection : projections) {
      network.set_projection_enabled(projection, enabled);
    }
  }

  Config config;
  SemanticNetwork network;
  std::uint32_t population_seed_offset{};

  SemanticPopulationId encoding{};
  SemanticPopulationId task{};
  SemanticPopulationId query{};
  SemanticPopulationId plan{};
  SemanticPopulationId transformation{};
  SemanticPopulationId action{};
  SemanticPopulationId decoding{};
  SemanticPopulationId motor{};
  std::vector<SemanticPopulationId> slots;

  std::vector<SemanticProjectionId> encoding_to_slots;
  SemanticProjectionId encoding_to_task{};
  SemanticProjectionId encoding_to_query{};
  SemanticProjectionId encoding_to_plan{};
  std::vector<SemanticProjectionId> encoding_projections;

  std::vector<SemanticProjectionId> slot_recurrence;
  SemanticProjectionId task_recurrence{};
  SemanticProjectionId query_recurrence{};
  SemanticProjectionId plan_recurrence{};
  std::vector<SemanticProjectionId> recurrence_projections;

  std::vector<SemanticProjectionId> slots_to_transformation;
  std::vector<SemanticProjectionId> slots_to_counting_start;
  std::vector<SemanticPopulationId> counting_stages;
  std::vector<SemanticProjectionId> counting_recurrence;
  std::vector<SemanticProjectionId> counting_successors;
  std::vector<SemanticProjectionId> counting_to_transformation;
  SemanticProjectionId plan_to_transformation{};
  std::vector<SemanticProjectionId> wm_to_transformation_projections;
  SemanticProjectionId transformation_to_action{};
  SemanticProjectionId action_to_decoding{};
  SemanticProjectionId decoding_to_motor{};

  std::array<bool, pathway_count> pathway_state{};
  std::vector<std::size_t> item_counts;
  std::size_t group_cursor{};
  std::size_t item_cursor{};
};

Workspace::Workspace(Config config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

Workspace::~Workspace() = default;
Workspace::Workspace(Workspace&&) noexcept = default;
Workspace& Workspace::operator=(Workspace&&) noexcept = default;

void Workspace::accept(const EncodedTokenEvent& event) {
  switch (event.kind) {
    case TokenKind::digit: {
      impl_->require_semantic(event);
      if (impl_->group_cursor >= impl_->config.maximum_groups) {
        throw std::overflow_error("Spaun workspace group capacity exceeded");
      }
      if (impl_->item_cursor >= impl_->config.maximum_list_length) {
        throw std::overflow_error("Spaun workspace list capacity exceeded");
      }
      const auto index =
          impl_->slot_index(impl_->group_cursor, impl_->item_cursor);
      impl_->write_population(impl_->encoding_to_slots[index],
                              impl_->slot_recurrence[index], *event.semantic);
      ++impl_->item_cursor;
      impl_->item_counts[impl_->group_cursor] =
          std::max(impl_->item_counts[impl_->group_cursor], impl_->item_cursor);
      break;
    }
    case TokenKind::task:
      impl_->require_semantic(event);
      impl_->write_population(impl_->encoding_to_task,
                              impl_->task_recurrence, *event.semantic);
      break;
    case TokenKind::query:
      impl_->require_semantic(event);
      impl_->write_population(impl_->encoding_to_query,
                              impl_->query_recurrence, *event.semantic);
      break;
    case TokenKind::group_boundary:
      if (event.semantic.has_value()) {
        throw std::invalid_argument(
            "group boundary event cannot contain a semantic value");
      }
      if (impl_->group_cursor >= impl_->config.maximum_groups) {
        throw std::overflow_error("Spaun workspace group capacity exceeded");
      }
      impl_->mark_recency(impl_->group_cursor, impl_->item_cursor);
      ++impl_->group_cursor;
      impl_->item_cursor = 0;
      break;
    case TokenKind::sequence_restart:
      if (event.semantic.has_value()) {
        throw std::invalid_argument(
            "sequence restart event cannot contain a semantic value");
      }
      impl_->group_cursor = 0;
      impl_->item_cursor = 0;
      std::fill(impl_->item_counts.begin(), impl_->item_counts.end(), 0);
      break;
  }
}

void Workspace::advance(const std::size_t steps) {
  for (std::size_t step = 0; step < steps; ++step) {
    static_cast<void>(impl_->network.step());
  }
}

std::optional<int> Workspace::recall_digit(const std::size_t group,
                                            const std::size_t item) const {
  const auto index = impl_->slot_index(group, item);
  if (item >= impl_->item_counts[group]) {
    return std::nullopt;
  }
  return impl_->cleanup_digit(impl_->slots[index]);
}

std::vector<int> Workspace::recall_group(const std::size_t group) const {
  if (group >= impl_->config.maximum_groups) {
    throw std::out_of_range("working-memory group is out of range");
  }
  std::vector<int> recalled;
  recalled.reserve(impl_->item_counts[group]);
  for (std::size_t item = 0; item < impl_->item_counts[group]; ++item) {
    const auto digit = recall_digit(group, item);
    if (!digit.has_value()) {
      return {};
    }
    recalled.push_back(*digit);
  }
  return recalled;
}

std::optional<int> Workspace::task_state() const {
  return impl_->cleanup_index(impl_->network.decode(impl_->task), task_names);
}

std::optional<char> Workspace::query_state() const {
  const auto query =
      impl_->cleanup_index(impl_->network.decode(impl_->query), query_names);
  if (!query.has_value()) {
    return std::nullopt;
  }
  return *query == 0 ? 'P' : 'K';
}

std::optional<int> Workspace::route_recalled_digit(const std::size_t group,
                                                    const std::size_t item) {
  const auto index = impl_->slot_index(group, item);
  if (item >= impl_->item_counts[group]) {
    return std::nullopt;
  }
  return impl_->cleanup_index(
      impl_->route(impl_->slots_to_transformation[index]), digit_names);
}

std::optional<int> Workspace::route_planned_digit(const Pointer& planned_digit) {
  if (planned_digit.dimension() != semantic_dimensions) {
    throw std::invalid_argument(
        "planned digit has the wrong semantic dimension");
  }
  impl_->write_population(impl_->encoding_to_plan,
                          impl_->plan_recurrence, planned_digit);
  return impl_->cleanup_index(impl_->route(impl_->plan_to_transformation),
                              digit_names);
}

bool Workspace::begin_counting_from_memory(const std::size_t group,
                                           const std::size_t item) {
  const auto index = impl_->slot_index(group, item);
  if (item >= impl_->item_counts[group]) {
    return false;
  }
  const auto route_enabled = impl_->pathway_state[pathway_index(
      Pathway::working_memory_to_transformation)];
  const auto selected_path = impl_->slots_to_counting_start[index];
  impl_->network.set_projection_enabled(selected_path, route_enabled);
  if (route_enabled) {
    impl_->network.enqueue_current_spikes(selected_path);
  }
  static_cast<void>(impl_->network.step());
  impl_->network.set_projection_enabled(selected_path, false);
  return impl_->cleanup_number(impl_->counting_stages.front()).has_value();
}

std::optional<int> Workspace::route_counting_result(
    const std::size_t increments) {
  if (increments >= impl_->counting_stages.size()) {
    throw std::out_of_range("counting increment is out of range");
  }
  return impl_->cleanup_index(
      impl_->route(impl_->counting_to_transformation[increments]),
      number_names);
}

std::optional<int> Workspace::counting_stage_state(
    const std::size_t increments) const {
  if (increments >= impl_->counting_stages.size()) {
    throw std::out_of_range("counting increment is out of range");
  }
  return impl_->cleanup_number(impl_->counting_stages[increments]);
}

std::size_t Workspace::counting_delay_ticks() const noexcept {
  return impl_->config.counting_delay_ticks;
}

void Workspace::set_pathway_enabled(const Pathway pathway,
                                    const bool enabled) {
  impl_->pathway_state[pathway_index(pathway)] = enabled;
  switch (pathway) {
    case Pathway::encoding_to_working_memory:
      // Encoding projections are pulse-gated during write events.  Disabling
      // every one here also clears any queued encoding drive immediately.
      if (!enabled) {
        impl_->set_all(impl_->encoding_projections, false);
      }
      break;
    case Pathway::working_memory_recurrence:
      impl_->set_all(impl_->recurrence_projections, enabled);
      if (!enabled) {
        // The projection setter clears delay queues; one zero-activity step
        // removes the final filtered spike so recall cannot see stale state.
        static_cast<void>(impl_->network.step());
      }
      break;
    case Pathway::working_memory_to_transformation:
      // These projections are likewise pulse-gated during routing.
      if (!enabled) {
        impl_->set_all(impl_->wm_to_transformation_projections, false);
      }
      break;
    case Pathway::transformation_action_to_decoding:
      impl_->network.set_projection_enabled(
          impl_->transformation_to_action, enabled);
      impl_->network.set_projection_enabled(impl_->action_to_decoding,
                                            enabled);
      break;
    case Pathway::decoding_to_motor:
      impl_->network.set_projection_enabled(impl_->decoding_to_motor, enabled);
      break;
  }
}

bool Workspace::pathway_enabled(const Pathway pathway) const {
  return impl_->pathway_state[pathway_index(pathway)];
}

void Workspace::reset() {
  impl_->network.reset_state();
  impl_->network.raw_network().restore_initial_weights();
  impl_->group_cursor = 0;
  impl_->item_cursor = 0;
  std::fill(impl_->item_counts.begin(), impl_->item_counts.end(), 0);

  impl_->set_all(impl_->encoding_projections, false);
  impl_->set_all(impl_->wm_to_transformation_projections, false);
  impl_->set_all(
      impl_->recurrence_projections,
      impl_->pathway_state[pathway_index(Pathway::working_memory_recurrence)]);
  const auto cognitive_output = impl_->pathway_state[pathway_index(
      Pathway::transformation_action_to_decoding)];
  impl_->network.set_projection_enabled(impl_->transformation_to_action,
                                        cognitive_output);
  impl_->network.set_projection_enabled(impl_->action_to_decoding,
                                        cognitive_output);
  impl_->network.set_projection_enabled(
      impl_->decoding_to_motor,
      impl_->pathway_state[pathway_index(Pathway::decoding_to_motor)]);
}

std::size_t Workspace::current_group() const noexcept {
  return impl_->group_cursor;
}

std::size_t Workspace::current_item() const noexcept {
  return impl_->item_cursor;
}

std::size_t Workspace::stored_item_count(const std::size_t group) const {
  if (group >= impl_->config.maximum_groups) {
    throw std::out_of_range("working-memory group is out of range");
  }
  return impl_->item_counts[group];
}

std::size_t Workspace::maximum_groups() const noexcept {
  return impl_->config.maximum_groups;
}

std::size_t Workspace::maximum_list_length() const noexcept {
  return impl_->config.maximum_list_length;
}

std::size_t Workspace::neuron_count() const noexcept {
  return impl_->network.neuron_count();
}

std::size_t Workspace::population_count() const noexcept {
  return impl_->network.population_count();
}

std::size_t Workspace::projection_count() const noexcept {
  return impl_->network.projection_count();
}

std::size_t Workspace::synapse_count() const noexcept {
  return impl_->network.synapse_count();
}

std::size_t Workspace::total_spikes() const noexcept {
  return impl_->network.raw_network().total_spikes();
}

std::uint64_t Workspace::time() const noexcept {
  return static_cast<std::uint64_t>(impl_->network.time());
}

}  // namespace snnbase_experiments::spaun::workspace
