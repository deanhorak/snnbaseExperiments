#include <snnbase_experiments/spaun.hpp>

#include <GL/freeglut.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using snnbase_experiments::spaun::Module;
using snnbase_experiments::spaun::ProbeFrame;
using snnbase_experiments::spaun::Result;
using snnbase_experiments::spaun::Task;

struct Options {
  Task task{Task::copy_drawing};
  std::uint32_t seed{42};
  int timer_ms{10};
  std::size_t exit_after_frames{};
  bool self_test{};
};

std::size_t parse_size(std::string_view value, std::string_view option,
                       bool allow_zero = false) {
  std::size_t parsed{};
  const auto [end, error] =
      std::from_chars(value.data(), value.data() + value.size(), parsed);
  if (error != std::errc{} || end != value.data() + value.size() ||
      (!allow_zero && parsed == 0)) {
    throw std::invalid_argument("invalid value for " + std::string(option));
  }
  return parsed;
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--self-test") {
      options.self_test = true;
      continue;
    }
    if (argument == "--help") {
      std::cout << "Usage: " << argv[0] << " [options]\n"
                << "  --task NAME       Initial A0-A7 task (default: A0)\n"
                << "  --seed N          Deterministic seed (default: 42)\n"
                << "  --timer-ms N      Playback timer interval (default: 10)\n"
                << "  --frames N        Exit after N rendered frames (GUI smoke test)\n"
                << "  --self-test       Exercise simulation/controller without opening a window\n";
      std::exit(EXIT_SUCCESS);
    }
    if (index + 1 >= argc) {
      throw std::invalid_argument("missing value for " + std::string(argument));
    }
    const std::string_view value(argv[++index]);
    if (argument == "--task") {
      options.task = snnbase_experiments::spaun::parse_task(value);
    } else if (argument == "--seed") {
      options.seed = static_cast<std::uint32_t>(parse_size(value, argument, true));
    } else if (argument == "--timer-ms") {
      options.timer_ms = static_cast<int>(parse_size(value, argument));
    } else if (argument == "--frames") {
      options.exit_after_frames = parse_size(value, argument);
    } else {
      throw std::invalid_argument("unknown option: " + std::string(argument));
    }
  }
  return options;
}

struct GuiState {
  explicit GuiState(const Options& options)
      : timer_ms(options.timer_ms), exit_after_frames(options.exit_after_frames) {
    snnbase_experiments::spaun::Model model({.seed = options.seed});
    for (const auto task : snnbase_experiments::spaun::all_tasks()) {
      results.push_back(
          model.run(snnbase_experiments::spaun::canonical_trial(task)));
    }
    task = static_cast<std::size_t>(options.task);
  }

  [[nodiscard]] const Result& result() const { return results.at(task); }
  [[nodiscard]] const ProbeFrame& frame() const {
    return result().frames.at(std::min(frame_index, result().frames.size() - 1));
  }

  std::vector<Result> results;
  std::size_t task{};
  std::size_t frame_index{};
  bool paused{};
  int timer_ms{10};
  std::size_t rendered_frames{};
  std::size_t exit_after_frames{};
  int width{1280};
  int height{760};
};

std::unique_ptr<GuiState> state;

void color(float red, float green, float blue, float alpha = 1.0F) {
  glColor4f(red, green, blue, alpha);
}

void rectangle(float x, float y, float width, float height, bool filled = true) {
  glBegin(filled ? GL_QUADS : GL_LINE_LOOP);
  glVertex2f(x, y);
  glVertex2f(x + width, y);
  glVertex2f(x + width, y + height);
  glVertex2f(x, y + height);
  glEnd();
}

void text(float x, float y, std::string_view value,
          void* font = GLUT_BITMAP_8_BY_13) {
  glRasterPos2f(x, y);
  for (const auto character : value) {
    glutBitmapCharacter(font, character);
  }
}

void panel(float x, float y, float width, float height, std::string_view title) {
  color(0.07F, 0.09F, 0.13F);
  rectangle(x, y, width, height);
  color(0.24F, 0.31F, 0.42F);
  rectangle(x, y, width, height, false);
  color(0.72F, 0.82F, 0.95F);
  text(x + 10.0F, y + height - 20.0F, title, GLUT_BITMAP_HELVETICA_12);
}

void draw_stimulus(const ProbeFrame& frame) {
  constexpr float x = 25.0F;
  constexpr float y = 460.0F;
  constexpr float width = 220.0F;
  constexpr float height = 270.0F;
  panel(x, y, width, height, "RETINA / 28 x 28 STIMULUS");
  const auto glyph = snnbase_experiments::spaun::glyph(frame.stimulus);
  constexpr float cell = 27.0F;
  const auto start_x = x + 42.0F;
  const auto start_y = y + 32.0F;
  for (std::size_t row = 0; row < glyph.size(); ++row) {
    for (std::size_t column = 0; column < glyph[row].size(); ++column) {
      const auto active = glyph[row][column] == '#';
      color(active ? 0.95F : 0.12F, active ? 0.96F : 0.14F,
            active ? 1.0F : 0.18F);
      rectangle(start_x + static_cast<float>(column) * cell,
                start_y + static_cast<float>(glyph.size() - 1 - row) * cell,
                cell - 2.0F, cell - 2.0F);
    }
  }
  color(0.70F, 0.78F, 0.88F);
  text(x + 12.0F, y + 10.0F,
       "decoded: " + std::string(1, frame.recognized));
}

std::array<float, 3> heat(double activity) {
  const auto bounded = static_cast<float>(std::clamp(activity, 0.0, 1.0));
  return {0.10F + 0.88F * bounded, 0.22F * (1.0F - bounded),
          0.72F * (1.0F - bounded)};
}

void circle(float x, float y, float radius) {
  constexpr int segments = 36;
  glBegin(GL_TRIANGLE_FAN);
  glVertex2f(x, y);
  for (int index = 0; index <= segments; ++index) {
    const auto angle = 2.0 * 3.14159265358979323846 *
                       static_cast<double>(index) / segments;
    glVertex2f(x + radius * static_cast<float>(std::cos(angle)),
               y + radius * static_cast<float>(std::sin(angle)));
  }
  glEnd();
}

void draw_brain(const ProbeFrame& frame) {
  constexpr float x = 265.0F;
  constexpr float y = 390.0F;
  constexpr float width = 545.0F;
  constexpr float height = 340.0F;
  panel(x, y, width, height, "FUNCTIONAL BRAIN MAP (BLUE LOW / RED HIGH)");
  constexpr std::array<std::array<float, 2>, snnbase_experiments::spaun::module_count>
      locations{{{55, 190}, {135, 230}, {240, 250}, {330, 225}, {255, 115},
                 {360, 115}, {420, 210}, {465, 125}, {490, 55}}};
  constexpr std::array<std::string_view,
                       snnbase_experiments::spaun::module_count>
      labels{"V1/V2/V4/IT", "AIT", "PPC/DLPFC", "VLPFC", "OFC/vStr",
             "Str/GPi", "medial PFC", "PM/SMA/M1", "ARM"};
  for (std::size_t module = 0; module < locations.size(); ++module) {
    const auto rgb = heat(frame.activity[module]);
    color(rgb[0], rgb[1], rgb[2]);
    circle(x + locations[module][0], y + locations[module][1], 27.0F);
    color(0.92F, 0.94F, 0.98F);
    text(x + locations[module][0] - 33.0F, y + locations[module][1] - 43.0F,
         labels[module], GLUT_BITMAP_HELVETICA_10);
  }
  color(0.58F, 0.66F, 0.78F);
  text(x + 12.0F, y + 12.0F, "phase: " + frame.phase);
}

void draw_activity(const ProbeFrame& frame) {
  constexpr float x = 25.0F;
  constexpr float y = 25.0F;
  constexpr float width = 405.0F;
  constexpr float height = 415.0F;
  panel(x, y, width, height, "MODULE RATES AND SPIKES");
  for (std::size_t module = 0; module < frame.activity.size(); ++module) {
    const auto row_y = y + height - 56.0F - static_cast<float>(module) * 39.0F;
    color(0.72F, 0.78F, 0.88F);
    text(x + 12.0F, row_y + 7.0F,
         snnbase_experiments::spaun::module_name(static_cast<Module>(module)));
    color(0.12F, 0.15F, 0.21F);
    rectangle(x + 145.0F, row_y, 210.0F, 18.0F);
    const auto rgb = heat(frame.activity[module]);
    color(rgb[0], rgb[1], rgb[2]);
    rectangle(x + 145.0F, row_y,
              210.0F * static_cast<float>(frame.activity[module]), 18.0F);
    color(0.82F, 0.86F, 0.93F);
    text(x + 362.0F, row_y + 5.0F, std::to_string(frame.spikes[module]));
  }
}

void draw_raster() {
  constexpr float x = 450.0F;
  constexpr float y = 25.0F;
  constexpr float width = 360.0F;
  constexpr float height = 340.0F;
  panel(x, y, width, height, "SPIKE RASTER (LAST 90 TICKS)");
  const auto& frames = state->result().frames;
  const auto end = std::min(state->frame_index + 1, frames.size());
  const auto begin = end > 90 ? end - 90 : 0;
  for (std::size_t module = 0; module < snnbase_experiments::spaun::module_count;
       ++module) {
    const auto row_y = y + 30.0F + static_cast<float>(module) * 30.0F;
    color(0.22F, 0.28F, 0.37F);
    glBegin(GL_LINES);
    glVertex2f(x + 35.0F, row_y);
    glVertex2f(x + width - 10.0F, row_y);
    glEnd();
    for (std::size_t index = begin; index < end; ++index) {
      if (frames[index].spikes[module] == 0) {
        continue;
      }
      const auto local = static_cast<float>(index - begin) / 90.0F;
      const auto spike_x = x + 35.0F + local * (width - 50.0F);
      const auto rgb = heat(frames[index].activity[module]);
      color(rgb[0], rgb[1], rgb[2]);
      glBegin(GL_LINES);
      glVertex2f(spike_x, row_y - 8.0F);
      glVertex2f(spike_x, row_y + 8.0F);
      glEnd();
    }
  }
}

void draw_arm(const ProbeFrame& frame) {
  constexpr float x = 830.0F;
  constexpr float y = 25.0F;
  constexpr float width = 425.0F;
  constexpr float height = 705.0F;
  panel(x, y, width, height, "TWO-JOINT ARM AND DRAWING SURFACE");
  const auto origin_x = x + width * 0.5F;
  const auto origin_y = y + 80.0F;
  constexpr float scale = 205.0F;
  color(0.19F, 0.22F, 0.29F);
  rectangle(x + 30.0F, y + 55.0F, width - 60.0F, 420.0F);
  color(0.22F, 0.85F, 0.72F);
  glLineWidth(2.0F);
  bool stroke_open = false;
  for (const auto& point : frame.pen_trace) {
    if (point.stroke_start || !stroke_open) {
      if (stroke_open) {
        glEnd();
      }
      glBegin(GL_LINE_STRIP);
      stroke_open = true;
    }
    glVertex2f(origin_x + static_cast<float>(point.x) * scale,
               origin_y + static_cast<float>(point.y) * scale);
  }
  if (stroke_open) {
    glEnd();
  } else {
    color(0.53F, 0.61F, 0.72F);
    if (frame.phase == "surface erase") {
      text(x + 65.0F, y + 265.0F,
           "surface erased - preparing next digit");
    } else if (frame.phase == "pen positioning") {
      text(x + 80.0F, y + 265.0F, "positioning drawing arm...");
    } else {
      text(x + 65.0F, y + 265.0F, "waiting for motor response...");
    }
  }
  color(0.90F, 0.65F, 0.23F);
  glLineWidth(8.0F);
  glBegin(GL_LINE_STRIP);
  glVertex2f(origin_x, origin_y);
  glVertex2f(origin_x + static_cast<float>(frame.arm.elbow_x) * scale,
             origin_y + static_cast<float>(frame.arm.elbow_y) * scale);
  glVertex2f(origin_x + static_cast<float>(frame.arm.pen_x) * scale,
             origin_y + static_cast<float>(frame.arm.pen_y) * scale);
  glEnd();
  glLineWidth(1.0F);
  color(frame.arm.pen_down ? 0.95F : 0.45F, 0.20F,
        frame.arm.pen_down ? 0.25F : 0.55F);
  circle(origin_x + static_cast<float>(frame.arm.pen_x) * scale,
         origin_y + static_cast<float>(frame.arm.pen_y) * scale, 7.0F);
  color(0.78F, 0.84F, 0.93F);
  text(x + 15.0F, y + 510.0F, "working memory: " + frame.working_memory);
  text(x + 15.0F, y + 535.0F, "selected route: " + frame.selected_action);
  text(x + 15.0F, y + 560.0F, "decoded output: " + frame.output);
  text(x + 15.0F, y + 585.0F,
       "reward: " + std::to_string(frame.reward).substr(0, 4));
  text(x + 15.0F, y + 625.0F, "CONTROLS");
  text(x + 15.0F, y + 645.0F,
       "space pause | n step | d jump to drawing | r reset");
  text(x + 15.0F, y + 665.0F,
       "0..7 task | +/- speed | q/esc quit");
}

void display() {
  glClear(GL_COLOR_BUFFER_BIT);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  const auto& result = state->result();
  const auto& frame = state->frame();
  color(0.88F, 0.93F, 1.0F);
  text(25.0F, 745.0F,
       "SNNBASE SPAUN  |  A" + std::to_string(static_cast<int>(result.task)) +
           " " + result.task_name + "  |  t=" +
           std::to_string(frame.time_seconds).substr(0, 5) + "s  |  " +
           (state->paused ? "PAUSED" : "RUNNING"),
       GLUT_BITMAP_HELVETICA_18);
  draw_stimulus(frame);
  draw_brain(frame);
  draw_activity(frame);
  draw_raster();
  draw_arm(frame);
  glutSwapBuffers();
  ++state->rendered_frames;
  if (state->exit_after_frames > 0 &&
      state->rendered_frames >= state->exit_after_frames) {
    glutLeaveMainLoop();
  }
}

void reshape(int width, int height) {
  state->width = width;
  state->height = height;
  glViewport(0, 0, width, height);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0.0, 1280.0, 0.0, 760.0, -1.0, 1.0);
  glMatrixMode(GL_MODELVIEW);
}

void timer(int) {
  if (!state->paused) {
    const auto last = state->result().frames.size() - 1;
    state->frame_index = std::min(state->frame_index + 1, last);
    if (state->frame_index == last) {
      state->paused = true;
    }
  }
  glutPostRedisplay();
  glutTimerFunc(static_cast<unsigned int>(state->timer_ms), timer, 0);
}

void keyboard(unsigned char key, int, int) {
  if (key >= '0' && key <= '7') {
    state->task = static_cast<std::size_t>(key - '0');
    state->frame_index = 0;
    state->paused = false;
  } else if (key == ' ') {
    state->paused = !state->paused;
  } else if (key == 'n' || key == 'N') {
    state->paused = true;
    state->frame_index =
        std::min(state->frame_index + 1, state->result().frames.size() - 1);
  } else if (key == 'r' || key == 'R') {
    state->frame_index = 0;
    state->paused = false;
  } else if (key == 'd' || key == 'D') {
    const auto& frames = state->result().frames;
    const auto drawing = std::find_if(
        frames.begin(), frames.end(), [](const ProbeFrame& frame) {
          return frame.phase == "pen positioning" ||
                 frame.phase == "motor execution";
        });
    if (drawing != frames.end()) {
      state->frame_index =
          static_cast<std::size_t>(std::distance(frames.begin(), drawing));
      state->paused = false;
    }
  } else if (key == '+' || key == '=') {
    state->timer_ms = std::max(5, state->timer_ms - 5);
  } else if (key == '-') {
    state->timer_ms = std::min(500, state->timer_ms + 5);
  } else if (key == 'q' || key == 'Q' || key == 27) {
    glutLeaveMainLoop();
  }
  glutPostRedisplay();
}

int self_test(const Options& options) {
  GuiState test_state(options);
  for (const auto& result : test_state.results) {
    const auto has_drawable_digit = std::any_of(
        result.output.begin(), result.output.end(),
        [](const int digit) { return digit >= 0 && digit <= 9; });
    if (result.frames.empty() || result.network_neurons == 0 ||
        result.network_synapses == 0 ||
        (has_drawable_digit &&
         (result.frames.back().pen_trace.empty() ||
          !result.frames.back().pen_trace.front().stroke_start))) {
      return EXIT_FAILURE;
    }
  }
  std::cout << "spaun_gui_self_test tasks=" << test_state.results.size()
            << " status=passed\n";
  return EXIT_SUCCESS;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto options = parse_options(argc, argv);
    if (options.self_test) {
      return self_test(options);
    }
    state = std::make_unique<GuiState>(options);
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_MULTISAMPLE);
    glutInitWindowSize(state->width, state->height);
    glutCreateWindow("snnbase Spaun simulation");
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
    glClearColor(0.025F, 0.035F, 0.055F, 1.0F);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(keyboard);
    glutTimerFunc(static_cast<unsigned int>(state->timer_ms), timer, 0);
    glutMainLoop();
    state.reset();
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "spaun_gui: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
