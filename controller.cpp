#define RAYGUI_IMPLEMENTATION
#include "raygui.h"
#if defined(PLATFORM_WEB)
#include <emscripten/emscripten.h>
#endif

#include "raylib.h"
#include "raymath.h"
#include "common.h"
#include "vec.h"
#include "quat.h"
#include "spring.h"
#include "array.h"
#include "character.h"
#include "database.h"
#include "nnet.h"
#include "lmm.h"

#include <initializer_list>
#include <functional>
#include <string>
#include <vector>

//--------------------------------------

static inline Vector3 to_Vector3(vec3 v)
{
    return (Vector3){ v.x, v.y, v.z };
}

static inline vec3 planar_velocity(vec3 v)
{
    return vec3(v.x, 0.0f, v.z);
}

static inline float planar_speed(vec3 v)
{
    return length(planar_velocity(v));
}

static inline vec3 clamp_planar_speed(vec3 v, float max_speed, const float eps = 1e-4f)
{
    vec3 planar = planar_velocity(v);
    float speed = length(planar);

    if (speed <= max_speed)
    {
        return planar;
    }

    return max_speed > eps ? normalize(planar) * max_speed : vec3();
}

static inline void damp_array_exact(
    slice1d<float> values,
    const slice1d<float> goals,
    const float halflife,
    const float dt)
{
    assert(values.size == goals.size);
    for (int i = 0; i < values.size; i++)
    {
        values(i) = damper_exact(values(i), goals(i), halflife, dt);
    }
}

struct environment_box
{
    vec3 position;
    vec3 scale;
    bool traversable;
    Color fill;
    Color wire;
};

struct environment_heightmap
{
    float x_min = 0.0f;
    float x_max = 0.0f;
    float z_min = 0.0f;
    float z_max = 0.0f;
    int width = 0;
    int height = 0;
    std::vector<float> data;

    bool loaded() const { return width > 1 && height > 1 && !data.empty(); }

    float sample(float x, float z) const
    {
        if (!loaded())
        {
            return 0.0f;
        }

        float px = (x - x_min) / maxf(x_max - x_min, 1e-5f) * (width - 1);
        float pz = (z - z_min) / maxf(z_max - z_min, 1e-5f) * (height - 1);

        int x0 = clamp((int)floorf(px), 0, width - 1);
        int x1 = clamp((int)ceilf(px), 0, width - 1);
        int z0 = clamp((int)floorf(pz), 0, height - 1);
        int z1 = clamp((int)ceilf(pz), 0, height - 1);

        float ax = px - floorf(px);
        float az = pz - floorf(pz);

        auto at = [&](int ix, int iz)
        {
            return data[ix + iz * width];
        };

        float s0 = at(x0, z0);
        float s1 = at(x1, z0);
        float s2 = at(x0, z1);
        float s3 = at(x1, z1);

        return lerpf(lerpf(s0, s1, ax), lerpf(s2, s3, ax), az);
    }

    bool load(const char* filename)
    {
        FILE* f = fopen(filename, "rb");
        if (f == NULL)
        {
            width = 0;
            height = 0;
            data.clear();
            return false;
        }

        fread(&width, sizeof(int), 1, f);
        fread(&height, sizeof(int), 1, f);
        fread(&x_min, sizeof(float), 1, f);
        fread(&x_max, sizeof(float), 1, f);
        fread(&z_min, sizeof(float), 1, f);
        fread(&z_max, sizeof(float), 1, f);

        if (width <= 1 || height <= 1)
        {
            fclose(f);
            width = 0;
            height = 0;
            data.clear();
            return false;
        }

        data.resize(width * height);
        fread(data.data(), sizeof(float), width * height, f);
        fclose(f);

        return true;
    }
};

static environment_heightmap g_environment_heightmap;
struct terrain_sampling_config_cpp
{
    std::vector<int> environment_horizons = { 10, 20, 40, 60, 80 };
    std::vector<int> matching_trajectory_horizons = { 20, 40, 60 };
    int prediction_frame_step = 10;
    float strip_half_width = 0.25f;
    float sdf_clamp_distance = 2.5f;
};

static terrain_sampling_config_cpp g_terrain_sampling_config;
struct eval_input_sample
{
    float left_x = 0.0f;
    float left_z = 0.0f;
    float right_x = 0.0f;
    float right_z = 0.0f;
    float gait = 0.0f;
    bool strafe = false;
};

struct eval_config_cpp
{
    bool enabled = false;
    bool use_learned = false;
    std::string script_path;
    std::string trace_path;
    std::string tag = "default";
    std::string selector_mode = "learned";
    std::string residual_mode = "learned";
    std::string failure_dump_dir;
    float start_x = 0.0f;
    float start_z = 0.0f;
    float start_yaw = 0.0f;
    int frames = 0;
};

static eval_config_cpp g_eval_config;
static std::vector<eval_input_sample> g_eval_script;

struct runtime_debug_snapshot
{
    int frame = 0;
    int teacher_frame = -1;
    int selected_exemplar = -1;
    bool exemplar_valid = false;
    bool switch_flag = false;
    bool divergence_flag = false;
    bool finite_query = true;
    bool finite_features = true;
    bool finite_latent = true;
    bool finite_residual = true;
    bool finite_stepper = true;
    bool finite_decompressor = true;
    bool finite_pose = true;
    float residual_norm_before = 0.0f;
    float residual_norm_after = 0.0f;
    float stepper_input_norm = 0.0f;
    float stepper_output_norm = 0.0f;
    float decompressor_input_norm = 0.0f;
    float root_planar_speed = 0.0f;
    float contact_slip = 0.0f;
    float correction_magnitude = 0.0f;
    float switch_severity = 0.0f;
    float residual_carry_alpha = 1.0f;
    float selector_margin = 0.0f;
    int exemplar_dwell_frames = 0;
    bool emergency_reanchor = false;
    bool hard_reset = false;
    std::vector<int> topk_indices;
    std::vector<float> topk_costs;
    std::vector<float> selector_scores;
    std::vector<float> query_normalized;
    std::vector<float> features_curr;
    std::vector<float> latent_curr;
    std::vector<float> residual_features;
    std::vector<float> residual_latent;
    std::vector<float> stepper_control;
};
static const char* PFNN_HEIGHTMAP_LABEL = "PFNN Rocky";
static const char* PFNN_TERRAIN_GRID_FILENAME = "./resources/pfnn_terrain_rocky_grid.bin";
static const char* PFNN_ENVIRONMENT_FEATURES_FILENAME = "resources/terrain_features.bin";
static const char* PFNN_ENVIRONMENT_BOXES_FILENAME = "./resources/environment_boxes.txt";
static const char* PFNN_TERRAIN_CONFIG_FILENAME = "./resources/terrain_sampling_config.txt";
static const int MOTION_MATCH_PHASE_CONTACT_FEATURE_COUNT = 4;
static const int MOTION_MATCH_BASE_FEATURE_COUNT = 31;
#if defined(PLATFORM_WEB)
static const char* CHECKERBOARD_VERTEX_SHADER_FILENAME = "./resources/checkerboard.vs";
static const char* CHECKERBOARD_FRAGMENT_SHADER_FILENAME = "./resources/checkerboard.fs";
static const char* CHARACTER_VERTEX_SHADER_FILENAME = "./resources/character.vs";
static const char* CHARACTER_FRAGMENT_SHADER_FILENAME = "./resources/character.fs";
#else
static const char* CHECKERBOARD_VERTEX_SHADER_FILENAME = "./resources/checkerboard_330.vs";
static const char* CHECKERBOARD_FRAGMENT_SHADER_FILENAME = "./resources/checkerboard_330.fs";
static const char* CHARACTER_VERTEX_SHADER_FILENAME = "./resources/character_330.vs";
static const char* CHARACTER_FRAGMENT_SHADER_FILENAME = "./resources/character_330.fs";
#endif
static bool load_environment_heightmap()
{
    return g_environment_heightmap.load(PFNN_TERRAIN_GRID_FILENAME);
}

static int terrain_feature_count()
{
    return 3 * (int)g_terrain_sampling_config.environment_horizons.size();
}

static int obstacle_sdf_feature_count()
{
    return terrain_feature_count();
}

static int control_feature_count(const int nfeatures)
{
    return std::max(nfeatures - 15, 0);
}

static const float HYBRID_RESIDUAL_NORM_P99 = 4.5259f;
static const float HYBRID_RESIDUAL_NORM_MAX = 32.0961f;
static const float HYBRID_STEPPER_INPUT_NORM_P99 = 25.1601f;
static const float HYBRID_STEPPER_INPUT_NORM_MAX = 72.4039f;
static const float HYBRID_FEATURE_DELTA_NORM_P95 = 2.7136f;
static const float HYBRID_CONTROL_DELTA_NORM_P95 = 2.4202f;
static const float HYBRID_PHASE_CONTACT_DELTA_NORM_P95 = 0.7789f;
static const float HYBRID_SWITCH_SOFT_SEVERITY = 2.0f;
static const float HYBRID_SWITCH_MEDIUM_SEVERITY = 3.5f;
static const float HYBRID_SWITCH_HARD_SEVERITY = 5.0f;
static const float HYBRID_CARRY_ALPHA_SOFT = 0.6f;
static const float HYBRID_CARRY_ALPHA_MEDIUM = 0.25f;
static const int HYBRID_SELECTOR_MIN_DWELL_FRAMES = 6;
static const float HYBRID_SELECTOR_SCORE_MARGIN = 4096.0f;
static const float HYBRID_CONTACT_SLIP_GUARD = 10.0f;
static const float HYBRID_CONTACT_SLIP_HARD_RESET = 25.0f;
static const int STARTUP_BOOTSTRAP_WINDOW_FRAMES = 8;
static const float STARTUP_SEVERE_PENETRATION = 0.05f;

static float slice_l2_distance(
    const slice1d<float> a,
    const slice1d<float> b,
    const int start,
    const int count)
{
    float accum = 0.0f;
    for (int i = 0; i < count; i++)
    {
        float delta = a(start + i) - b(start + i);
        accum += delta * delta;
    }
    return sqrtf(accum);
}

static float contact_state_mismatch(
    const slice1d<bool> contacts,
    const slice1d<bool> exemplar_contacts)
{
    int mismatches = 0;
    int count = std::min(contacts.size, exemplar_contacts.size);
    for (int i = 0; i < count; i++)
    {
        mismatches += contacts(i) != exemplar_contacts(i) ? 1 : 0;
    }
    return count > 0 ? (float)mismatches / (float)count : 0.0f;
}

static float hybrid_switch_severity(
    const slice1d<float> features_curr,
    const slice1d<float> exemplar_features,
    const slice1d<bool> curr_contacts,
    const slice1d<bool> exemplar_contacts)
{
    float feature_delta = slice_l2_distance(features_curr, exemplar_features, 0, features_curr.size);
    float control_delta = slice_l2_distance(features_curr, exemplar_features, 15, control_feature_count(features_curr.size));
    float phase_contact_delta = slice_l2_distance(features_curr, exemplar_features, 27, MOTION_MATCH_PHASE_CONTACT_FEATURE_COUNT);
    float contact_mismatch = contact_state_mismatch(curr_contacts, exemplar_contacts);

    return
        feature_delta / HYBRID_FEATURE_DELTA_NORM_P95 +
        0.5f * (control_delta / HYBRID_CONTROL_DELTA_NORM_P95) +
        0.5f * (phase_contact_delta / HYBRID_PHASE_CONTACT_DELTA_NORM_P95) +
        1.5f * contact_mismatch;
}

static float hybrid_residual_norm(
    const slice1d<float> residual_features,
    const slice1d<float> residual_latent)
{
    float residual_feature_norm = 0.0f;
    float residual_latent_norm = 0.0f;
    for (int i = 0; i < residual_features.size; i++)
    {
        residual_feature_norm += squaref(residual_features(i));
    }
    for (int i = 0; i < residual_latent.size; i++)
    {
        residual_latent_norm += squaref(residual_latent(i));
    }
    return sqrtf(
        residual_feature_norm +
        residual_latent_norm);
}

static void hybrid_reconstruct_absolute_state(
    slice1d<float> features_curr,
    slice1d<float> latent_curr,
    const slice1d<float> anchor_features,
    const slice1d<float> anchor_latent,
    const slice1d<float> residual_features,
    const slice1d<float> residual_latent)
{
    for (int i = 0; i < features_curr.size; i++)
    {
        features_curr(i) = anchor_features(i) + residual_features(i);
    }
    for (int i = 0; i < latent_curr.size; i++)
    {
        latent_curr(i) = anchor_latent(i) + residual_latent(i);
    }
}

static int prediction_sample_count()
{
    int max_horizon = 0;
    for (int horizon : g_terrain_sampling_config.environment_horizons)
    {
        max_horizon = std::max(max_horizon, horizon);
    }
    for (int horizon : g_terrain_sampling_config.matching_trajectory_horizons)
    {
        max_horizon = std::max(max_horizon, horizon);
    }
    return 1 + max_horizon / std::max(g_terrain_sampling_config.prediction_frame_step, 1);
}

static int prediction_index_for_horizon(const int horizon)
{
    return horizon / std::max(g_terrain_sampling_config.prediction_frame_step, 1);
}

static int environment_feature_offset_for_horizon(const int horizon)
{
    for (int i = 0; i < (int)g_terrain_sampling_config.environment_horizons.size(); i++)
    {
        if (g_terrain_sampling_config.environment_horizons[i] == horizon)
        {
            return i * 3;
        }
    }
    return -1;
}

static bool parse_int_list(std::vector<int>& values, const std::string& text)
{
    values.clear();
    size_t start = 0;
    while (start < text.size())
    {
        size_t end = text.find(',', start);
        std::string token = text.substr(start, end == std::string::npos ? std::string::npos : end - start);
        if (!token.empty())
        {
            values.push_back(atoi(token.c_str()));
        }
        if (end == std::string::npos)
        {
            break;
        }
        start = end + 1;
        while (start < text.size() && text[start] == ' ')
        {
            start++;
        }
    }
    return !values.empty();
}

static bool load_terrain_sampling_config(const char* filename)
{
    FILE* f = fopen(filename, "r");
    if (f == NULL)
    {
        return false;
    }

    terrain_sampling_config_cpp config;
    char line[256];
    while (fgets(line, sizeof(line), f) != NULL)
    {
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r')
        {
            continue;
        }

        char* eq = strchr(line, '=');
        if (eq == NULL)
        {
            continue;
        }

        *eq = '\0';
        std::string key(line);
        std::string value(eq + 1);
        while (!value.empty() && (value.back() == '\n' || value.back() == '\r'))
        {
            value.pop_back();
        }

        if (key == "environment_horizons")
        {
            parse_int_list(config.environment_horizons, value);
        }
        else if (key == "matching_trajectory_horizons")
        {
            parse_int_list(config.matching_trajectory_horizons, value);
        }
        else if (key == "prediction_frame_step")
        {
            config.prediction_frame_step = atoi(value.c_str());
        }
        else if (key == "strip_half_width")
        {
            config.strip_half_width = atof(value.c_str());
        }
        else if (key == "sdf_clamp_distance")
        {
            config.sdf_clamp_distance = atof(value.c_str());
        }
    }

    fclose(f);

    if (config.environment_horizons.empty() || config.matching_trajectory_horizons.size() != 3 || config.prediction_frame_step <= 0)
    {
        return false;
    }

    g_terrain_sampling_config = config;
    return true;
}

static bool file_exists(const char* filename)
{
    FILE* f = fopen(filename, "rb");
    if (f == NULL)
    {
        return false;
    }
    fclose(f);
    return true;
}

static float array_norm(const slice1d<float> values)
{
    float total = 0.0f;
    for (int i = 0; i < values.size; i++)
    {
        total += squaref(values(i));
    }
    return sqrtf(total);
}

static bool array_is_finite(const slice1d<float> values)
{
    for (int i = 0; i < values.size; i++)
    {
        if (!isfinite(values(i)))
        {
            return false;
        }
    }
    return true;
}

static bool vec3_array_is_finite(const slice1d<vec3> values)
{
    for (int i = 0; i < values.size; i++)
    {
        if (!isfinite(values(i).x) || !isfinite(values(i).y) || !isfinite(values(i).z))
        {
            return false;
        }
    }
    return true;
}

static std::vector<float> copy_array(const slice1d<float> values)
{
    std::vector<float> out(values.size);
    for (int i = 0; i < values.size; i++)
    {
        out[i] = values(i);
    }
    return out;
}

static std::vector<int> copy_array_int(const slice1d<int> values)
{
    std::vector<int> out(values.size);
    for (int i = 0; i < values.size; i++)
    {
        out[i] = values(i);
    }
    return out;
}

static void write_debug_vector(FILE* f, const char* label, const std::vector<float>& values)
{
    fprintf(f, "%s:", label);
    for (float value : values)
    {
        fprintf(f, " %.9g", value);
    }
    fprintf(f, "\n");
}

static void write_debug_vector_int(FILE* f, const char* label, const std::vector<int>& values)
{
    fprintf(f, "%s:", label);
    for (int value : values)
    {
        fprintf(f, " %d", value);
    }
    fprintf(f, "\n");
}

static void write_runtime_snapshot(FILE* f, const runtime_debug_snapshot& snapshot)
{
    fprintf(
        f,
        "frame=%d teacher_frame=%d selected_exemplar=%d exemplar_valid=%d switch=%d divergence=%d "
        "finite_query=%d finite_features=%d finite_latent=%d finite_residual=%d finite_stepper=%d finite_decompressor=%d finite_pose=%d "
        "residual_norm_before=%.9g residual_norm_after=%.9g stepper_input_norm=%.9g stepper_output_norm=%.9g "
        "decompressor_input_norm=%.9g root_planar_speed=%.9g contact_slip=%.9g correction_magnitude=%.9g "
        "switch_severity=%.9g residual_carry_alpha=%.9g selector_margin=%.9g exemplar_dwell_frames=%d emergency_reanchor=%d hard_reset=%d\n",
        snapshot.frame,
        snapshot.teacher_frame,
        snapshot.selected_exemplar,
        snapshot.exemplar_valid ? 1 : 0,
        snapshot.switch_flag ? 1 : 0,
        snapshot.divergence_flag ? 1 : 0,
        snapshot.finite_query ? 1 : 0,
        snapshot.finite_features ? 1 : 0,
        snapshot.finite_latent ? 1 : 0,
        snapshot.finite_residual ? 1 : 0,
        snapshot.finite_stepper ? 1 : 0,
        snapshot.finite_decompressor ? 1 : 0,
        snapshot.finite_pose ? 1 : 0,
        snapshot.residual_norm_before,
        snapshot.residual_norm_after,
        snapshot.stepper_input_norm,
        snapshot.stepper_output_norm,
        snapshot.decompressor_input_norm,
        snapshot.root_planar_speed,
        snapshot.contact_slip,
        snapshot.correction_magnitude,
        snapshot.switch_severity,
        snapshot.residual_carry_alpha,
        snapshot.selector_margin,
        snapshot.exemplar_dwell_frames,
        snapshot.emergency_reanchor ? 1 : 0,
        snapshot.hard_reset ? 1 : 0);

    write_debug_vector_int(f, "topk_indices", snapshot.topk_indices);
    write_debug_vector(f, "topk_costs", snapshot.topk_costs);
    write_debug_vector(f, "selector_scores", snapshot.selector_scores);
    write_debug_vector(f, "query_normalized", snapshot.query_normalized);
    write_debug_vector(f, "features_curr", snapshot.features_curr);
    write_debug_vector(f, "latent_curr", snapshot.latent_curr);
    write_debug_vector(f, "residual_features", snapshot.residual_features);
    write_debug_vector(f, "residual_latent", snapshot.residual_latent);
    write_debug_vector(f, "stepper_control", snapshot.stepper_control);
    fprintf(f, "---\n");
}

static bool load_eval_script(const char* filename, std::vector<eval_input_sample>& samples)
{
    FILE* f = fopen(filename, "r");
    if (f == NULL)
    {
        return false;
    }

    samples.clear();
    char line[256];
    while (fgets(line, sizeof(line), f) != NULL)
    {
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r')
        {
            continue;
        }

        eval_input_sample sample;
        int strafe = 0;
        int parsed = sscanf(
            line,
            " %f %f %f %f %f %d",
            &sample.left_x,
            &sample.left_z,
            &sample.right_x,
            &sample.right_z,
            &sample.gait,
            &strafe);

        if (parsed == 6)
        {
            sample.strafe = strafe != 0;
            samples.push_back(sample);
        }
    }

    fclose(f);
    return !samples.empty();
}

static bool eval_parse_args(int argc, char** argv)
{
    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];
        auto require_value = [&](const char* name) -> const char*
        {
            assert(i + 1 < argc);
            i++;
            return argv[i];
        };

        if (arg == "--eval-script")
        {
            g_eval_config.enabled = true;
            g_eval_config.script_path = require_value("--eval-script");
        }
        else if (arg == "--eval-trace")
        {
            g_eval_config.trace_path = require_value("--eval-trace");
        }
        else if (arg == "--eval-tag")
        {
            g_eval_config.tag = require_value("--eval-tag");
        }
        else if (arg == "--eval-mode")
        {
            g_eval_config.enabled = true;
            std::string value = require_value("--eval-mode");
            g_eval_config.use_learned = value == "learned";
        }
        else if (arg == "--hybrid-selector-mode")
        {
            g_eval_config.selector_mode = require_value("--hybrid-selector-mode");
        }
        else if (arg == "--hybrid-residual-mode")
        {
            g_eval_config.residual_mode = require_value("--hybrid-residual-mode");
        }
        else if (arg == "--failure-dump-dir")
        {
            g_eval_config.failure_dump_dir = require_value("--failure-dump-dir");
        }
        else if (arg == "--eval-start-x")
        {
            g_eval_config.start_x = atof(require_value("--eval-start-x"));
        }
        else if (arg == "--eval-start-z")
        {
            g_eval_config.start_z = atof(require_value("--eval-start-z"));
        }
        else if (arg == "--eval-start-yaw")
        {
            g_eval_config.start_yaw = atof(require_value("--eval-start-yaw"));
        }
        else if (arg == "--eval-frames")
        {
            g_eval_config.frames = atoi(require_value("--eval-frames"));
        }
    }

    if (!g_eval_config.enabled)
    {
        return true;
    }

    if (g_eval_config.script_path.empty() || g_eval_config.trace_path.empty() || g_eval_config.frames <= 0)
    {
        return false;
    }

    return load_eval_script(g_eval_config.script_path.c_str(), g_eval_script);
}

float environment_base_height(float x, float z)
{
    return g_environment_heightmap.sample(x, z);
}

static environment_box make_environment_box(vec3 position, vec3 scale, bool traversable)
{
    environment_box box;
    box.position = position;
    box.scale = scale;
    box.traversable = traversable;
    box.fill = traversable ? Color{ 90, 136, 173, 110 } : Color{ 147, 96, 74, 120 };
    box.wire = traversable ? Color{ 53, 87, 114, 255 } : Color{ 96, 54, 41, 255 };
    return box;
}

static void load_default_environment_boxes(array1d<environment_box>& environment_boxes)
{
    environment_boxes.resize(3);
    environment_boxes(0) = make_environment_box(vec3(1.75f, 0.0f, 1.25f), vec3(1.2f, 1.0f, 2.2f), false);
    environment_boxes(1) = make_environment_box(vec3(-1.35f, 0.0f, -1.10f), vec3(1.8f, 1.0f, 1.6f), false);
    environment_boxes(2) = make_environment_box(vec3(0.15f, 0.0f, 2.05f), vec3(1.1f, 1.0f, 1.0f), false);
}

static bool load_environment_boxes(array1d<environment_box>& environment_boxes, const char* filename)
{
    FILE* f = fopen(filename, "r");
    if (f == NULL)
    {
        return false;
    }

    std::vector<environment_box> boxes;
    char line[256];

    while (fgets(line, sizeof(line), f) != NULL)
    {
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r')
        {
            continue;
        }

        float px, py, pz;
        float sx, sy, sz;
        int traversable = 0;
        int parsed = sscanf(line, " %f %f %f %f %f %f %d", &px, &py, &pz, &sx, &sy, &sz, &traversable);

        if (parsed == 7)
        {
            boxes.push_back(make_environment_box(
                vec3(px, py, pz),
                vec3(sx, sy, sz),
                traversable != 0));
        }
    }

    fclose(f);

    environment_boxes.resize((int)boxes.size());
    for (int i = 0; i < environment_boxes.size; i++)
    {
        environment_boxes(i) = boxes[i];
    }

    return true;
}

static void initialize_environment_boxes(array1d<environment_box>& environment_boxes)
{
    if (!load_environment_boxes(environment_boxes, PFNN_ENVIRONMENT_BOXES_FILENAME))
    {
        load_default_environment_boxes(environment_boxes);
    }
}

static inline bool environment_box_contains_xz(
    const environment_box& box,
    const float x,
    const float z,
    const float padding = 0.0f)
{
    return
        x >= box.position.x - 0.5f * box.scale.x - padding &&
        x <= box.position.x + 0.5f * box.scale.x + padding &&
        z >= box.position.z - 0.5f * box.scale.z - padding &&
        z <= box.position.z + 0.5f * box.scale.z + padding;
}

static inline float environment_box_base(const environment_box& box)
{
    return environment_base_height(box.position.x, box.position.z) + box.position.y;
}

static inline float environment_box_top(const environment_box& box)
{
    return environment_box_base(box) + box.scale.y;
}

static inline float environment_box_signed_distance_xz(
    const environment_box& box,
    const float x,
    const float z)
{
    float dx = fabsf(x - box.position.x) - 0.5f * box.scale.x;
    float dz = fabsf(z - box.position.z) - 0.5f * box.scale.z;

    float outside_dx = maxf(dx, 0.0f);
    float outside_dz = maxf(dz, 0.0f);
    float outside_distance = sqrtf(outside_dx * outside_dx + outside_dz * outside_dz);
    float inside_distance = minf(maxf(dx, dz), 0.0f);

    return outside_distance + inside_distance;
}

float environment_obstacle_sdf(
    const float x,
    const float z,
    const slice1d<environment_box> environment_boxes,
    const float clamp_distance = g_terrain_sampling_config.sdf_clamp_distance)
{
    float sdf = clamp_distance;
    bool found_obstacle = false;

    for (int i = 0; i < environment_boxes.size; i++)
    {
        const environment_box& box = environment_boxes(i);

        if (box.traversable)
        {
            continue;
        }

        sdf = minf(sdf, environment_box_signed_distance_xz(box, x, z));
        found_obstacle = true;
    }

    return found_obstacle ? clampf(sdf, -clamp_distance, clamp_distance) : clamp_distance;
}

float environment_surface_height(
    const float x,
    const float z,
    const slice1d<environment_box> environment_boxes)
{
    float height = environment_base_height(x, z);

    for (int i = 0; i < environment_boxes.size; i++)
    {
        const environment_box& box = environment_boxes(i);

        if (!box.traversable)
        {
            continue;
        }

        if (environment_box_contains_xz(box, x, z))
        {
            height = maxf(height, environment_box_top(box));
        }
    }

    return height;
}

vec3 environment_surface_normal(
    const float x,
    const float z,
    const slice1d<environment_box> environment_boxes,
    const float eps = 0.05f)
{
    float hx0 = environment_surface_height(x - eps, z, environment_boxes);
    float hx1 = environment_surface_height(x + eps, z, environment_boxes);
    float hz0 = environment_surface_height(x, z - eps, environment_boxes);
    float hz1 = environment_surface_height(x, z + eps, environment_boxes);
    vec3 normal = normalize(vec3(-(hx1 - hx0), 2.0f * eps, -(hz1 - hz0)));
    return length(normal) > 1e-5f ? normal : vec3(0.0f, 1.0f, 0.0f);
}

void terrain_mesh_apply_heightfield(Mesh& mesh, bool upload_to_gpu = false)
{
    for (int i = 0; i < mesh.vertexCount; i++)
    {
        float x = mesh.vertices[3 * i + 0];
        float z = mesh.vertices[3 * i + 2];

        mesh.vertices[3 * i + 1] = environment_base_height(x, z);
        mesh.normals[3 * i + 0] = 0.0f;
        mesh.normals[3 * i + 1] = 0.0f;
        mesh.normals[3 * i + 2] = 0.0f;
    }

    for (int i = 0; i < mesh.triangleCount; i++)
    {
        unsigned short i0 = mesh.indices[3 * i + 0];
        unsigned short i1 = mesh.indices[3 * i + 1];
        unsigned short i2 = mesh.indices[3 * i + 2];

        vec3 v0(
            mesh.vertices[3 * i0 + 0],
            mesh.vertices[3 * i0 + 1],
            mesh.vertices[3 * i0 + 2]);
        vec3 v1(
            mesh.vertices[3 * i1 + 0],
            mesh.vertices[3 * i1 + 1],
            mesh.vertices[3 * i1 + 2]);
        vec3 v2(
            mesh.vertices[3 * i2 + 0],
            mesh.vertices[3 * i2 + 1],
            mesh.vertices[3 * i2 + 2]);

        vec3 normal = normalize(cross(v1 - v0, v2 - v0));

        mesh.normals[3 * i0 + 0] += normal.x;
        mesh.normals[3 * i0 + 1] += normal.y;
        mesh.normals[3 * i0 + 2] += normal.z;

        mesh.normals[3 * i1 + 0] += normal.x;
        mesh.normals[3 * i1 + 1] += normal.y;
        mesh.normals[3 * i1 + 2] += normal.z;

        mesh.normals[3 * i2 + 0] += normal.x;
        mesh.normals[3 * i2 + 1] += normal.y;
        mesh.normals[3 * i2 + 2] += normal.z;
    }

    for (int i = 0; i < mesh.vertexCount; i++)
    {
        vec3 normal(
            mesh.normals[3 * i + 0],
            mesh.normals[3 * i + 1],
            mesh.normals[3 * i + 2]);

        normal = normalize(normal);

        mesh.normals[3 * i + 0] = normal.x;
        mesh.normals[3 * i + 1] = normal.y;
        mesh.normals[3 * i + 2] = normal.z;
    }

    if (upload_to_gpu)
    {
        UpdateMeshBuffer(mesh, 0, mesh.vertices, mesh.vertexCount * 3 * sizeof(float), 0);
        UpdateMeshBuffer(mesh, 2, mesh.normals, mesh.vertexCount * 3 * sizeof(float), 0);
    }
}

void terrain_mesh_offset_xz(Mesh& mesh, const float offset_x, const float offset_z)
{
    for (int i = 0; i < mesh.vertexCount; i++)
    {
        mesh.vertices[3 * i + 0] += offset_x;
        mesh.vertices[3 * i + 2] += offset_z;
    }
}

//--------------------------------------

// Perform linear blend skinning and copy 
// result into mesh data. Update and upload 
// deformed vertex positions and normals to GPU
void deform_character_mesh(
  Mesh& mesh, 
  const character& c,
  const slice1d<vec3> bone_anim_positions,
  const slice1d<quat> bone_anim_rotations,
  const slice1d<int> bone_parents)
{
    linear_blend_skinning_positions(
        slice1d<vec3>(mesh.vertexCount, (vec3*)mesh.vertices),
        c.positions,
        c.bone_weights,
        c.bone_indices,
        c.bone_rest_positions,
        c.bone_rest_rotations,
        bone_anim_positions,
        bone_anim_rotations);
    
    linear_blend_skinning_normals(
        slice1d<vec3>(mesh.vertexCount, (vec3*)mesh.normals),
        c.normals,
        c.bone_weights,
        c.bone_indices,
        c.bone_rest_rotations,
        bone_anim_rotations);
    
    UpdateMeshBuffer(mesh, 0, mesh.vertices, mesh.vertexCount * 3 * sizeof(float), 0);
    UpdateMeshBuffer(mesh, 2, mesh.normals, mesh.vertexCount * 3 * sizeof(float), 0);
}

Mesh make_character_mesh(character& c)
{
    Mesh mesh = { 0 };
    
    mesh.vertexCount = c.positions.size;
    mesh.triangleCount = c.triangles.size / 3;
    mesh.vertices = (float*)MemAlloc(c.positions.size * 3 * sizeof(float));
    mesh.texcoords = (float*)MemAlloc(c.texcoords.size * 2 * sizeof(float));
    mesh.normals = (float*)MemAlloc(c.normals.size * 3 * sizeof(float));
    mesh.indices = (unsigned short*)MemAlloc(c.triangles.size * sizeof(unsigned short));
    
    memcpy(mesh.vertices, c.positions.data, c.positions.size * 3 * sizeof(float));
    memcpy(mesh.texcoords, c.texcoords.data, c.texcoords.size * 2 * sizeof(float));
    memcpy(mesh.normals, c.normals.data, c.normals.size * 3 * sizeof(float));
    memcpy(mesh.indices, c.triangles.data, c.triangles.size * sizeof(unsigned short));
    
    UploadMesh(&mesh, true);
    
    return mesh;
}

//--------------------------------------

// Basic functionality to get gamepad input including deadzone and 
// squaring of the stick location to increase sensitivity. To make 
// all the other code that uses this easier, we assume stick is 
// oriented on floor (i.e. y-axis is zero)

enum
{
    GAMEPAD_PLAYER = 0,
};

enum
{
    GAMEPAD_STICK_LEFT,
    GAMEPAD_STICK_RIGHT,
};

vec3 gamepad_get_stick(int stick, const float deadzone = 0.2f)
{
    float gamepadx = GetGamepadAxisMovement(GAMEPAD_PLAYER, stick == GAMEPAD_STICK_LEFT ? GAMEPAD_AXIS_LEFT_X : GAMEPAD_AXIS_RIGHT_X);
    float gamepady = GetGamepadAxisMovement(GAMEPAD_PLAYER, stick == GAMEPAD_STICK_LEFT ? GAMEPAD_AXIS_LEFT_Y : GAMEPAD_AXIS_RIGHT_Y);

    if (stick == GAMEPAD_STICK_LEFT)
    {
        if (IsKeyDown(KEY_A)) gamepadx -= 1.0f;
        if (IsKeyDown(KEY_D)) gamepadx += 1.0f;
        if (IsKeyDown(KEY_W)) gamepady -= 1.0f;
        if (IsKeyDown(KEY_S)) gamepady += 1.0f;
    }
    else if (stick == GAMEPAD_STICK_RIGHT)
    {
        if (IsKeyDown(KEY_LEFT))  gamepadx -= 1.0f;
        if (IsKeyDown(KEY_RIGHT)) gamepadx += 1.0f;
        if (IsKeyDown(KEY_UP))    gamepady -= 1.0f;
        if (IsKeyDown(KEY_DOWN))  gamepady += 1.0f;

        if (IsMouseButtonDown(MOUSE_BUTTON_RIGHT))
        {
            Vector2 mouse_delta = GetMouseDelta();
            const float mouse_sensitivity = 0.045f;
            gamepadx += mouse_sensitivity * mouse_delta.x;
            gamepady += mouse_sensitivity * mouse_delta.y;
        }
    }

    float gamepadmag = sqrtf(gamepadx*gamepadx + gamepady*gamepady);
    
    if (gamepadmag > deadzone)
    {
        float gamepaddirx = gamepadx / gamepadmag;
        float gamepaddiry = gamepady / gamepadmag;
        float gamepadclippedmag = gamepadmag > 1.0f ? 1.0f : gamepadmag*gamepadmag;
        gamepadx = gamepaddirx * gamepadclippedmag;
        gamepady = gamepaddiry * gamepadclippedmag;
    }
    else
    {
        gamepadx = 0.0f;
        gamepady = 0.0f;
    }
    
    return vec3(gamepadx, 0.0f, gamepady);
}

//--------------------------------------

float orbit_camera_update_azimuth(
    const float azimuth, 
    const vec3 gamepadstick_right,
    const bool desired_strafe,
    const float dt)
{
    vec3 gamepadaxis = desired_strafe ? vec3() : gamepadstick_right;
    return azimuth + 2.0f * dt * -gamepadaxis.x;
}

float orbit_camera_update_altitude(
    const float altitude, 
    const vec3 gamepadstick_right,
    const bool desired_strafe,
    const float dt)
{
    vec3 gamepadaxis = desired_strafe ? vec3() : gamepadstick_right;
    return clampf(altitude + 2.0f * dt * gamepadaxis.z, 0.0, 0.4f * PIf);
}

float orbit_camera_update_distance(
    const float distance, 
    const float dt)
{
    float gamepadzoom = 
        IsGamepadButtonDown(GAMEPAD_PLAYER, GAMEPAD_BUTTON_LEFT_TRIGGER_1)  ? +1.0f :
        IsGamepadButtonDown(GAMEPAD_PLAYER, GAMEPAD_BUTTON_RIGHT_TRIGGER_1) ? -1.0f : 0.0f;
        
    return clampf(distance +  10.0f * dt * gamepadzoom, 0.1f, 100.0f);
}

// Updates the camera using the orbit cam controls
void orbit_camera_update(
    Camera3D& cam, 
    float& camera_azimuth,
    float& camera_altitude,
    float& camera_distance,
    const vec3 target,
    const vec3 gamepadstick_right,
    const bool desired_strafe,
    const float dt)
{
    camera_azimuth = orbit_camera_update_azimuth(camera_azimuth, gamepadstick_right, desired_strafe, dt);
    camera_altitude = orbit_camera_update_altitude(camera_altitude, gamepadstick_right, desired_strafe, dt);
    camera_distance = orbit_camera_update_distance(camera_distance, dt);
    
    quat rotation_azimuth = quat_from_angle_axis(camera_azimuth, vec3(0, 1, 0));
    vec3 position = quat_mul_vec3(rotation_azimuth, vec3(0, 0, camera_distance));
    vec3 axis = normalize(cross(position, vec3(0, 1, 0)));
    
    quat rotation_altitude = quat_from_angle_axis(camera_altitude, axis);
    
    vec3 eye = target + quat_mul_vec3(rotation_altitude, position);

    cam.target = (Vector3){ target.x, target.y, target.z };
    cam.position = (Vector3){ eye.x, eye.y, eye.z };
}

//--------------------------------------

bool desired_strafe_update()
{
    return IsGamepadButtonDown(GAMEPAD_PLAYER, GAMEPAD_BUTTON_LEFT_TRIGGER_2) > 0.5f;
}

void desired_gait_update(
    float& desired_gait, 
    float& desired_gait_velocity,
    const float dt,
    const float gait_change_halflife = 0.1f)
{
    simple_spring_damper_exact(
        desired_gait, 
        desired_gait_velocity,
        IsGamepadButtonDown(GAMEPAD_PLAYER, GAMEPAD_BUTTON_RIGHT_FACE_DOWN) ? 1.0f : 0.0f,
        gait_change_halflife,
        dt);
}

vec3 desired_velocity_update(
    const vec3 gamepadstick_left,
    const float camera_azimuth,
    const quat simulation_rotation,
    const float fwrd_speed,
    const float side_speed,
    const float back_speed)
{
    // Find stick position in world space by rotating using camera azimuth
    vec3 global_stick_direction = quat_mul_vec3(
        quat_from_angle_axis(camera_azimuth, vec3(0, 1, 0)), gamepadstick_left);
    
    // Find stick position local to current facing direction
    vec3 local_stick_direction = quat_inv_mul_vec3(
        simulation_rotation, global_stick_direction);
    
    // Scale stick by forward, sideways and backwards speeds
    vec3 local_desired_velocity = local_stick_direction.z > 0.0 ?
        vec3(side_speed, 0.0f, fwrd_speed) * local_stick_direction :
        vec3(side_speed, 0.0f, back_speed) * local_stick_direction;
    
    // Re-orientate into the world space
    return quat_mul_vec3(simulation_rotation, local_desired_velocity);
}

vec3 simulation_desired_velocity_update(
    const vec3 desired_velocity,
    const vec3 animation_velocity,
    const bool follow_animation_speed,
    const float max_animation_speed,
    const float input_deadzone = 0.01f,
    const float speed_epsilon = 1e-4f)
{
    if (!follow_animation_speed)
    {
        return desired_velocity;
    }

    vec3 desired_planar_velocity = planar_velocity(desired_velocity);
    vec3 animation_planar_velocity = clamp_planar_speed(animation_velocity, max_animation_speed);
    float animation_root_speed = planar_speed(animation_velocity);
    animation_root_speed = minf(animation_root_speed, max_animation_speed);

    if (length(desired_planar_velocity) > input_deadzone)
    {
        return animation_root_speed > speed_epsilon ?
            normalize(desired_planar_velocity) * animation_root_speed :
            vec3();
    }

    return animation_root_speed > speed_epsilon ? animation_planar_velocity : vec3();
}

void locomotion_request_speeds_update(
    float& fwrd_speed,
    float& side_speed,
    float& back_speed,
    const float desired_gait,
    const bool follow_animation_speed,
    const float manual_run_fwrd_speed,
    const float manual_run_side_speed,
    const float manual_run_back_speed,
    const float manual_walk_fwrd_speed,
    const float manual_walk_side_speed,
    const float manual_walk_back_speed)
{
    if (follow_animation_speed)
    {
        // Keep follow mode inside the speed range this database actually uses
        // so the query asks for locomotion intent, not arbitrary root speeds.
        const float follow_run_fwrd_speed = 2.2f;
        const float follow_run_side_speed = 1.7f;
        const float follow_run_back_speed = 1.35f;
        const float follow_walk_fwrd_speed = 1.1f;
        const float follow_walk_side_speed = 0.9f;
        const float follow_walk_back_speed = 0.75f;

        fwrd_speed = lerpf(follow_run_fwrd_speed, follow_walk_fwrd_speed, desired_gait);
        side_speed = lerpf(follow_run_side_speed, follow_walk_side_speed, desired_gait);
        back_speed = lerpf(follow_run_back_speed, follow_walk_back_speed, desired_gait);
        return;
    }

    fwrd_speed = lerpf(manual_run_fwrd_speed, manual_walk_fwrd_speed, desired_gait);
    side_speed = lerpf(manual_run_side_speed, manual_walk_side_speed, desired_gait);
    back_speed = lerpf(manual_run_back_speed, manual_walk_back_speed, desired_gait);
}

quat desired_rotation_update(
    const quat desired_rotation,
    const vec3 gamepadstick_left,
    const vec3 gamepadstick_right,
    const float camera_azimuth,
    const bool desired_strafe,
    const vec3 desired_velocity)
{
    quat desired_rotation_curr = desired_rotation;
    
    // If strafe is active then desired direction is coming from right
    // stick as long as that stick is being used, otherwise we assume
    // forward facing
    if (desired_strafe)
    {
        vec3 desired_direction = quat_mul_vec3(quat_from_angle_axis(camera_azimuth, vec3(0, 1, 0)), vec3(0, 0, -1));

        if (length(gamepadstick_right) > 0.01f)
        {
            desired_direction = quat_mul_vec3(quat_from_angle_axis(camera_azimuth, vec3(0, 1, 0)), normalize(gamepadstick_right));
        }
        
        return quat_from_angle_axis(atan2f(desired_direction.x, desired_direction.z), vec3(0, 1, 0));            
    }
    
    // If strafe is not active the desired direction comes from the left 
    // stick as long as that stick is being used
    else if (length(gamepadstick_left) > 0.01f)
    {
        
        vec3 desired_direction = normalize(desired_velocity);
        return quat_from_angle_axis(atan2f(desired_direction.x, desired_direction.z), vec3(0, 1, 0));
    }
    
    // Otherwise desired direction remains the same
    else
    {
        return desired_rotation_curr;
    }
}

//--------------------------------------

// Moving the root is a little bit difficult when we have the
// inertializer set up in the way we do. Essentially we need
// to also make sure to adjust all of the locations where 
// we are transforming the data to and from as well as the 
// offsets being blended out
void inertialize_root_adjust(
    vec3& offset_position,
    vec3& transition_src_position,
    quat& transition_src_rotation,
    vec3& transition_dst_position,
    quat& transition_dst_rotation,
    vec3& position,
    quat& rotation,
    const vec3 input_position,
    const quat input_rotation)
{
    // Find the position difference and add it to the state and transition location
    vec3 position_difference = input_position - position;
    position = position_difference + position;
    transition_dst_position = position_difference + transition_dst_position;
    
    // Find the point at which we want to now transition from in the src data
    transition_src_position = transition_src_position + quat_mul_vec3(transition_src_rotation,
        quat_inv_mul_vec3(transition_dst_rotation, position - offset_position - transition_dst_position));
    transition_dst_position = position;
    offset_position = vec3();
    
    // Find the rotation difference. We need to normalize here or some error can accumulate 
    // over time during adjustment.
    quat rotation_difference = quat_normalize(quat_mul_inv(input_rotation, rotation));
    
    // Apply the rotation difference to the current rotation and transition location
    rotation = quat_mul(rotation_difference, rotation);
    transition_dst_rotation = quat_mul(rotation_difference, transition_dst_rotation);
}

void inertialize_pose_reset(
    slice1d<vec3> bone_offset_positions,
    slice1d<vec3> bone_offset_velocities,
    slice1d<quat> bone_offset_rotations,
    slice1d<vec3> bone_offset_angular_velocities,
    vec3& transition_src_position,
    quat& transition_src_rotation,
    vec3& transition_dst_position,
    quat& transition_dst_rotation,
    const vec3 root_position,
    const quat root_rotation)
{
    bone_offset_positions.zero();
    bone_offset_velocities.zero();
    bone_offset_rotations.set(quat());
    bone_offset_angular_velocities.zero();
    
    transition_src_position = root_position;
    transition_src_rotation = root_rotation;
    transition_dst_position = vec3();
    transition_dst_rotation = quat();
}

// This function transitions the inertializer for 
// the full character. It takes as input the current 
// offsets, as well as the root transition locations,
// current root state, and the full pose information 
// for the pose being transitioned from (src) as well 
// as the pose being transitioned to (dst) in their
// own animation spaces.
void inertialize_pose_transition(
    slice1d<vec3> bone_offset_positions,
    slice1d<vec3> bone_offset_velocities,
    slice1d<quat> bone_offset_rotations,
    slice1d<vec3> bone_offset_angular_velocities,
    vec3& transition_src_position,
    quat& transition_src_rotation,
    vec3& transition_dst_position,
    quat& transition_dst_rotation,
    const vec3 root_position,
    const vec3 root_velocity,
    const quat root_rotation,
    const vec3 root_angular_velocity,
    const slice1d<vec3> bone_src_positions,
    const slice1d<vec3> bone_src_velocities,
    const slice1d<quat> bone_src_rotations,
    const slice1d<vec3> bone_src_angular_velocities,
    const slice1d<vec3> bone_dst_positions,
    const slice1d<vec3> bone_dst_velocities,
    const slice1d<quat> bone_dst_rotations,
    const slice1d<vec3> bone_dst_angular_velocities)
{
    // First we record the root position and rotation
    // in the animation data for the source and destination
    // animation
    transition_dst_position = root_position;
    transition_dst_rotation = root_rotation;
    transition_src_position = bone_dst_positions(0);
    transition_src_rotation = bone_dst_rotations(0);
    
    // We then find the velocities so we can transition the 
    // root inertiaizers
    vec3 world_space_dst_velocity = quat_mul_vec3(transition_dst_rotation, 
        quat_inv_mul_vec3(transition_src_rotation, bone_dst_velocities(0)));
    
    vec3 world_space_dst_angular_velocity = quat_mul_vec3(transition_dst_rotation, 
        quat_inv_mul_vec3(transition_src_rotation, bone_dst_angular_velocities(0)));
    
    // Transition inertializers recording the offsets for 
    // the root joint
    inertialize_transition(
        bone_offset_positions(0),
        bone_offset_velocities(0),
        root_position,
        root_velocity,
        root_position,
        world_space_dst_velocity);
        
    inertialize_transition(
        bone_offset_rotations(0),
        bone_offset_angular_velocities(0),
        root_rotation,
        root_angular_velocity,
        root_rotation,
        world_space_dst_angular_velocity);
    
    // Transition all the inertializers for each other bone
    for (int i = 1; i < bone_offset_positions.size; i++)
    {
        inertialize_transition(
            bone_offset_positions(i),
            bone_offset_velocities(i),
            bone_src_positions(i),
            bone_src_velocities(i),
            bone_dst_positions(i),
            bone_dst_velocities(i));
            
        inertialize_transition(
            bone_offset_rotations(i),
            bone_offset_angular_velocities(i),
            bone_src_rotations(i),
            bone_src_angular_velocities(i),
            bone_dst_rotations(i),
            bone_dst_angular_velocities(i));
    }
}

// This function updates the inertializer states. Here 
// it outputs the smoothed animation (input plus offset) 
// as well as updating the offsets themselves. It takes 
// as input the current playing animation as well as the 
// root transition locations, a halflife, and a dt
void inertialize_pose_update(
    slice1d<vec3> bone_positions,
    slice1d<vec3> bone_velocities,
    slice1d<quat> bone_rotations,
    slice1d<vec3> bone_angular_velocities,
    slice1d<vec3> bone_offset_positions,
    slice1d<vec3> bone_offset_velocities,
    slice1d<quat> bone_offset_rotations,
    slice1d<vec3> bone_offset_angular_velocities,
    const slice1d<vec3> bone_input_positions,
    const slice1d<vec3> bone_input_velocities,
    const slice1d<quat> bone_input_rotations,
    const slice1d<vec3> bone_input_angular_velocities,
    const vec3 transition_src_position,
    const quat transition_src_rotation,
    const vec3 transition_dst_position,
    const quat transition_dst_rotation,
    const float halflife,
    const float dt)
{
    // First we find the next root position, velocity, rotation
    // and rotational velocity in the world space by transforming 
    // the input animation from it's animation space into the 
    // space of the currently playing animation.
    vec3 world_space_position = quat_mul_vec3(transition_dst_rotation, 
        quat_inv_mul_vec3(transition_src_rotation, 
            bone_input_positions(0) - transition_src_position)) + transition_dst_position;
    
    vec3 world_space_velocity = quat_mul_vec3(transition_dst_rotation, 
        quat_inv_mul_vec3(transition_src_rotation, bone_input_velocities(0)));
    
    // Normalize here because quat inv mul can sometimes produce 
    // unstable returns when the two rotations are very close.
    quat world_space_rotation = quat_normalize(quat_mul(transition_dst_rotation, 
        quat_inv_mul(transition_src_rotation, bone_input_rotations(0))));
    
    vec3 world_space_angular_velocity = quat_mul_vec3(transition_dst_rotation, 
        quat_inv_mul_vec3(transition_src_rotation, bone_input_angular_velocities(0)));
    
    // Then we update these two inertializers with these new world space inputs
    inertialize_update(
        bone_positions(0),
        bone_velocities(0),
        bone_offset_positions(0),
        bone_offset_velocities(0),
        world_space_position,
        world_space_velocity,
        halflife,
        dt);
        
    inertialize_update(
        bone_rotations(0),
        bone_angular_velocities(0),
        bone_offset_rotations(0),
        bone_offset_angular_velocities(0),
        world_space_rotation,
        world_space_angular_velocity,
        halflife,
        dt);        
    
    // Then we update the inertializers for the rest of the bones
    for (int i = 1; i < bone_positions.size; i++)
    {
        inertialize_update(
            bone_positions(i),
            bone_velocities(i),
            bone_offset_positions(i),
            bone_offset_velocities(i),
            bone_input_positions(i),
            bone_input_velocities(i),
            halflife,
            dt);
            
        inertialize_update(
            bone_rotations(i),
            bone_angular_velocities(i),
            bone_offset_rotations(i),
            bone_offset_angular_velocities(i),
            bone_input_rotations(i),
            bone_input_angular_velocities(i),
            halflife,
            dt);
    }
}

//--------------------------------------

// Copy a part of a feature vector from the 
// matching database into the query feature vector
void query_copy_denormalized_feature(
    slice1d<float> query, 
    int& offset, 
    const int size, 
    const slice1d<float> features,
    const slice1d<float> features_offset,
    const slice1d<float> features_scale)
{
    for (int i = 0; i < size; i++)
    {
        query(offset + i) = features(offset + i) * features_scale(offset + i) + features_offset(offset + i);
    }
    
    offset += size;
}

// Compute the query feature vector for the current 
// trajectory controlled by the gamepad.
void query_compute_trajectory_position_feature(
    slice1d<float> query, 
    int& offset, 
    const vec3 root_position, 
    const quat root_rotation, 
    const slice1d<vec3> trajectory_positions)
{
    assert(g_terrain_sampling_config.matching_trajectory_horizons.size() == 3);
    int idx0 = prediction_index_for_horizon(g_terrain_sampling_config.matching_trajectory_horizons[0]);
    int idx1 = prediction_index_for_horizon(g_terrain_sampling_config.matching_trajectory_horizons[1]);
    int idx2 = prediction_index_for_horizon(g_terrain_sampling_config.matching_trajectory_horizons[2]);
    vec3 traj0 = quat_inv_mul_vec3(root_rotation, trajectory_positions(idx0) - root_position);
    vec3 traj1 = quat_inv_mul_vec3(root_rotation, trajectory_positions(idx1) - root_position);
    vec3 traj2 = quat_inv_mul_vec3(root_rotation, trajectory_positions(idx2) - root_position);
    
    query(offset + 0) = traj0.x;
    query(offset + 1) = traj0.z;
    query(offset + 2) = traj1.x;
    query(offset + 3) = traj1.z;
    query(offset + 4) = traj2.x;
    query(offset + 5) = traj2.z;
    
    offset += 6;
}

// Same but for the trajectory direction
void query_compute_trajectory_direction_feature(
    slice1d<float> query, 
    int& offset, 
    const quat root_rotation, 
    const slice1d<quat> trajectory_rotations)
{
    assert(g_terrain_sampling_config.matching_trajectory_horizons.size() == 3);
    int idx0 = prediction_index_for_horizon(g_terrain_sampling_config.matching_trajectory_horizons[0]);
    int idx1 = prediction_index_for_horizon(g_terrain_sampling_config.matching_trajectory_horizons[1]);
    int idx2 = prediction_index_for_horizon(g_terrain_sampling_config.matching_trajectory_horizons[2]);
    vec3 traj0 = quat_inv_mul_vec3(root_rotation, quat_mul_vec3(trajectory_rotations(idx0), vec3(0, 0, 1)));
    vec3 traj1 = quat_inv_mul_vec3(root_rotation, quat_mul_vec3(trajectory_rotations(idx1), vec3(0, 0, 1)));
    vec3 traj2 = quat_inv_mul_vec3(root_rotation, quat_mul_vec3(trajectory_rotations(idx2), vec3(0, 0, 1)));
    
    query(offset + 0) = traj0.x;
    query(offset + 1) = traj0.z;
    query(offset + 2) = traj1.x;
    query(offset + 3) = traj1.z;
    query(offset + 4) = traj2.x;
    query(offset + 5) = traj2.z;
    
    offset += 6;
}

void query_compute_environment_feature(
    slice1d<float> query,
    int& offset,
    const vec3 root_position,
    const slice1d<vec3> trajectory_positions,
    const slice1d<quat> trajectory_rotations,
    const slice1d<environment_box> environment_boxes,
    const int environment_feature_count)
{
    const float root_ground_height = environment_base_height(root_position.x, root_position.z);
    int terrain_offset = offset;

    for (int horizon : g_terrain_sampling_config.environment_horizons)
    {
        int trajectory_index = prediction_index_for_horizon(horizon);
        vec3 center = trajectory_positions(trajectory_index);
        vec3 right = quat_mul_vec3(trajectory_rotations(trajectory_index), vec3(1.0f, 0.0f, 0.0f));

        vec3 sample_right = center + g_terrain_sampling_config.strip_half_width * right;
        vec3 sample_left = center - g_terrain_sampling_config.strip_half_width * right;

        if (environment_feature_count >= terrain_feature_count())
        {
            query(terrain_offset + 0) = environment_base_height(sample_right.x, sample_right.z) - root_ground_height;
            query(terrain_offset + 1) = environment_base_height(center.x, center.z) - root_ground_height;
            query(terrain_offset + 2) = environment_base_height(sample_left.x, sample_left.z) - root_ground_height;
        }

        if (environment_feature_count >= terrain_feature_count() + obstacle_sdf_feature_count())
        {
            int sdf_offset = offset + terrain_feature_count() + (terrain_offset - offset);
            query(sdf_offset + 0) = environment_obstacle_sdf(sample_right.x, sample_right.z, environment_boxes);
            query(sdf_offset + 1) = environment_obstacle_sdf(center.x, center.z, environment_boxes);
            query(sdf_offset + 2) = environment_obstacle_sdf(sample_left.x, sample_left.z, environment_boxes);
        }

        terrain_offset += 3;
    }

    for (int i = terrain_feature_count() + obstacle_sdf_feature_count(); i < environment_feature_count; i++)
    {
        query(offset + i) = 0.0f;
    }

    offset += environment_feature_count;
}

//--------------------------------------

// Collide against the obscales which are
// essentially bounding boxes of a given size
vec3 simulation_collide_obstacles(
    const vec3 prev_pos,
    const vec3 next_pos,
    const slice1d<environment_box> environment_boxes,
    const float radius = 0.6f)
{
    vec3 dx = next_pos - prev_pos;
    vec3 proj_pos = prev_pos;
    proj_pos.y = environment_surface_height(proj_pos.x, proj_pos.z, environment_boxes);
    
    // Substep because I'm too lazy to implement CCD
    int substeps = 1 + (int)(length(dx) * 5.0f);
    
    for (int j = 0; j < substeps; j++)
    {
        proj_pos = proj_pos + dx / substeps;
        proj_pos.y = environment_surface_height(proj_pos.x, proj_pos.z, environment_boxes);
        
        for (int i = 0; i < environment_boxes.size; i++)
        {
            const environment_box& box = environment_boxes(i);

            if (box.traversable)
            {
                continue;
            }

            if (proj_pos.y > environment_box_top(box) + 0.4f)
            {
                continue;
            }

            float nearest_x = clampf(
                proj_pos.x,
                box.position.x - 0.5f * box.scale.x,
                box.position.x + 0.5f * box.scale.x);
            float nearest_z = clampf(
                proj_pos.z,
                box.position.z - 0.5f * box.scale.z,
                box.position.z + 0.5f * box.scale.z);

            vec3 delta = vec3(proj_pos.x - nearest_x, 0.0f, proj_pos.z - nearest_z);
            float delta_length = length(delta);

            if (delta_length < radius)
            {
                if (delta_length < 1e-5f)
                {
                    float dist_left = fabsf(proj_pos.x - (box.position.x - 0.5f * box.scale.x));
                    float dist_right = fabsf((box.position.x + 0.5f * box.scale.x) - proj_pos.x);
                    float dist_back = fabsf(proj_pos.z - (box.position.z - 0.5f * box.scale.z));
                    float dist_front = fabsf((box.position.z + 0.5f * box.scale.z) - proj_pos.z);

                    if (minf(dist_left, dist_right) < minf(dist_back, dist_front))
                    {
                        delta = vec3(dist_left < dist_right ? -1.0f : 1.0f, 0.0f, 0.0f);
                    }
                    else
                    {
                        delta = vec3(0.0f, 0.0f, dist_back < dist_front ? -1.0f : 1.0f);
                    }

                    delta_length = 1.0f;
                }

                delta = delta / delta_length;
                proj_pos = vec3(
                    nearest_x + radius * delta.x,
                    proj_pos.y,
                    nearest_z + radius * delta.z);
            }
        }
    } 
    
    proj_pos.y = environment_surface_height(proj_pos.x, proj_pos.z, environment_boxes);
    return proj_pos;
}

// Taken from https://theorangeduck.com/page/spring-roll-call#controllers
void simulation_positions_update(
    vec3& position, 
    vec3& velocity, 
    vec3& acceleration, 
    const vec3 desired_velocity, 
    const float halflife, 
    const float dt,
    const slice1d<environment_box> environment_boxes)
{
    float y = halflife_to_damping(halflife) / 2.0f; 
    vec3 j0 = velocity - desired_velocity;
    vec3 j1 = acceleration + j0*y;
    float eydt = fast_negexpf(y*dt);
    
    vec3 position_prev = position;

    position = eydt*(((-j1)/(y*y)) + ((-j0 - j1*dt)/y)) + 
        (j1/(y*y)) + j0/y + desired_velocity * dt + position_prev;
    velocity = eydt*(j0 + j1*dt) + desired_velocity;
    acceleration = eydt*(acceleration - j1*y*dt);
    
    position = simulation_collide_obstacles(
        position_prev, 
        position,
        environment_boxes);
}

void simulation_rotations_update(
    quat& rotation, 
    vec3& angular_velocity, 
    const quat desired_rotation, 
    const float halflife, 
    const float dt)
{
    simple_spring_damper_exact(
        rotation, 
        angular_velocity, 
        desired_rotation, 
        halflife, dt);
}

// Predict what the desired velocity will be in the 
// future. Here we need to use the future trajectory 
// rotation as well as predicted future camera 
// position to find an accurate desired velocity in 
// the world space
void trajectory_desired_velocities_predict(
  slice1d<vec3> desired_velocities,
  const slice1d<quat> trajectory_rotations,
  const vec3 desired_velocity,
  const float camera_azimuth,
  const vec3 gamepadstick_left,
  const vec3 gamepadstick_right,
  const bool desired_strafe,
  const float fwrd_speed,
  const float side_speed,
  const float back_speed,
  const float dt)
{
    desired_velocities(0) = desired_velocity;
    
    for (int i = 1; i < desired_velocities.size; i++)
    {
        desired_velocities(i) = desired_velocity_update(
            gamepadstick_left,
            orbit_camera_update_azimuth(
                camera_azimuth, gamepadstick_right, desired_strafe, i * dt),
            trajectory_rotations(i),
            fwrd_speed,
            side_speed,
            back_speed);
    }
}

void trajectory_positions_predict(
    slice1d<vec3> positions, 
    slice1d<vec3> velocities, 
    slice1d<vec3> accelerations, 
    const vec3 position, 
    const vec3 velocity, 
    const vec3 acceleration, 
    const slice1d<vec3> desired_velocities, 
    const float halflife,
    const float dt,
    const slice1d<environment_box> environment_boxes)
{
    positions(0) = position;
    velocities(0) = velocity;
    accelerations(0) = acceleration;
    
    for (int i = 1; i < positions.size; i++)
    {
        positions(i) = positions(i-1);
        velocities(i) = velocities(i-1);
        accelerations(i) = accelerations(i-1);
        
        simulation_positions_update(
            positions(i), 
            velocities(i), 
            accelerations(i), 
            desired_velocities(i), 
            halflife, 
            dt, 
            environment_boxes);
    }
}

// Predict desired rotations given the estimated future 
// camera rotation and other parameters
void trajectory_desired_rotations_predict(
  slice1d<quat> desired_rotations,
  const slice1d<vec3> desired_velocities,
  const quat desired_rotation,
  const float camera_azimuth,
  const vec3 gamepadstick_left,
  const vec3 gamepadstick_right,
  const bool desired_strafe,
  const float dt)
{
    desired_rotations(0) = desired_rotation;
    
    for (int i = 1; i < desired_rotations.size; i++)
    {
        desired_rotations(i) = desired_rotation_update(
            desired_rotations(i-1),
            gamepadstick_left,
            gamepadstick_right,
            orbit_camera_update_azimuth(
                camera_azimuth, gamepadstick_right, desired_strafe, i * dt),
            desired_strafe,
            desired_velocities(i));
    }
}

void trajectory_rotations_predict(
    slice1d<quat> rotations, 
    slice1d<vec3> angular_velocities, 
    const quat rotation, 
    const vec3 angular_velocity, 
    const slice1d<quat> desired_rotations, 
    const float halflife,
    const float dt)
{
    rotations.set(rotation);
    angular_velocities.set(angular_velocity);
    
    for (int i = 1; i < rotations.size; i++)
    {
        simulation_rotations_update(
            rotations(i), 
            angular_velocities(i), 
            desired_rotations(i), 
            halflife, 
            i * dt);
    }
}

//--------------------------------------

void contact_reset(
    bool& contact_state,
    bool& contact_lock,
    vec3& contact_position,
    vec3& contact_velocity,
    vec3& contact_point,
    vec3& contact_target,
    vec3& contact_offset_position,
    vec3& contact_offset_velocity,
    const vec3 input_contact_position,
    const vec3 input_contact_velocity,
    const bool input_contact_state)
{
    contact_state = false;
    contact_lock = false;
    contact_position = input_contact_position;
    contact_velocity = input_contact_velocity;
    contact_point = input_contact_position;
    contact_target = input_contact_position;
    contact_offset_position = vec3();
    contact_offset_velocity = vec3();
}

void contact_bootstrap(
    bool& contact_state,
    bool& contact_lock,
    vec3& contact_position,
    vec3& contact_velocity,
    vec3& contact_point,
    vec3& contact_target,
    vec3& contact_offset_position,
    vec3& contact_offset_velocity,
    const vec3 input_contact_position,
    const bool input_contact_state,
    const float ground_height,
    const float foot_height)
{
    contact_state = input_contact_state;
    contact_lock = input_contact_state;
    contact_position = input_contact_position;
    contact_position.y = ground_height + foot_height;
    contact_velocity = vec3();
    contact_point = contact_position;
    contact_target = input_contact_position;
    contact_offset_position = vec3();
    contact_offset_velocity = vec3();
}

void contact_update(
    bool& contact_state,
    bool& contact_lock,
    vec3& contact_position,
    vec3& contact_velocity,
    vec3& contact_point,
    vec3& contact_target,
    vec3& contact_offset_position,
    vec3& contact_offset_velocity,
    const vec3 input_contact_position,
    const bool input_contact_state,
    const float unlock_radius,
    const float ground_height,
    const float foot_height,
    const float halflife,
    const float dt,
    const float eps=1e-8)
{
    // First compute the input contact position velocity via finite difference
    vec3 input_contact_velocity = 
        (input_contact_position - contact_target) / (dt + eps);    
    contact_target = input_contact_position;
    
    // Update the inertializer to tick forward in time
    inertialize_update(
        contact_position,
        contact_velocity,
        contact_offset_position,
        contact_offset_velocity,
        // If locked we feed the contact point and zero velocity, 
        // otherwise we feed the input from the animation
        contact_lock ? contact_point : input_contact_position,
        contact_lock ?        vec3() : input_contact_velocity,
        halflife,
        dt);
    
    // If the contact point is too far from the current input position 
    // then we need to unlock the contact
    bool unlock_contact = contact_lock && (
        length(contact_point - input_contact_position) > unlock_radius);
    
    // If the contact was previously inactive but is now active we 
    // need to transition to the locked contact state
    if (!contact_state && input_contact_state)
    {
        // Contact point is given by the current position of 
        // the foot projected onto the ground plus foot height
        contact_lock = true;
        contact_point = contact_position;
        contact_point.y = ground_height + foot_height;
        
        inertialize_transition(
            contact_offset_position,
            contact_offset_velocity,
            input_contact_position,
            input_contact_velocity,
            contact_point,
            vec3());
    }
    
    // Otherwise if we need to unlock or we were previously in 
    // contact but are no longer we transition to just taking 
    // the input position as-is
    else if ((contact_lock && contact_state && !input_contact_state) 
         || unlock_contact)
    {
        contact_lock = false;
        
        inertialize_transition(
            contact_offset_position,
            contact_offset_velocity,
            contact_point,
            vec3(),
            input_contact_position,
            input_contact_velocity);
    }
    
    // Update contact state
    contact_state = input_contact_state;
}

//--------------------------------------

// Rotate a joint to look toward some 
// given target position
void ik_look_at(
    quat& bone_rotation,
    const quat global_parent_rotation,
    const quat global_rotation,
    const vec3 global_position,
    const vec3 child_position,
    const vec3 target_position,
    const float eps = 1e-5f)
{
    vec3 curr_dir = normalize(child_position - global_position);
    vec3 targ_dir = normalize(target_position - global_position);

    if (fabs(1.0f - dot(curr_dir, targ_dir) > eps))
    {
        bone_rotation = quat_inv_mul(global_parent_rotation, 
            quat_mul(quat_between(curr_dir, targ_dir), global_rotation));
    }
}

// Basic two-joint IK in the style of https://theorangeduck.com/page/simple-two-joint
// Here I add a basic "forward vector" which acts like a kind of pole-vetor
// to control the bending direction
void ik_two_bone(
    quat& bone_root_lr, 
    quat& bone_mid_lr,
    const vec3 bone_root, 
    const vec3 bone_mid, 
    const vec3 bone_end, 
    const vec3 target, 
    const vec3 fwd,
    const quat bone_root_gr, 
    const quat bone_mid_gr,
    const quat bone_par_gr,
    const float max_length_buffer) {
    
    float max_extension = 
        length(bone_root - bone_mid) + 
        length(bone_mid - bone_end) - 
        max_length_buffer;
    
    vec3 target_clamp = target;
    if (length(target - bone_root) > max_extension)
    {
        target_clamp = bone_root + max_extension * normalize(target - bone_root);
    }
    
    vec3 axis_dwn = normalize(bone_end - bone_root);
    vec3 axis_rot = normalize(cross(axis_dwn, fwd));

    vec3 a = bone_root;
    vec3 b = bone_mid;
    vec3 c = bone_end;
    vec3 t = target_clamp;
    
    float lab = length(b - a);
    float lcb = length(b - c);
    float lat = length(t - a);

    float ac_ab_0 = acosf(clampf(dot(normalize(c - a), normalize(b - a)), -1.0f, 1.0f));
    float ba_bc_0 = acosf(clampf(dot(normalize(a - b), normalize(c - b)), -1.0f, 1.0f));

    float ac_ab_1 = acosf(clampf((lab * lab + lat * lat - lcb * lcb) / (2.0f * lab * lat), -1.0f, 1.0f));
    float ba_bc_1 = acosf(clampf((lab * lab + lcb * lcb - lat * lat) / (2.0f * lab * lcb), -1.0f, 1.0f));

    quat r0 = quat_from_angle_axis(ac_ab_1 - ac_ab_0, axis_rot);
    quat r1 = quat_from_angle_axis(ba_bc_1 - ba_bc_0, axis_rot);

    vec3 c_a = normalize(bone_end - bone_root);
    vec3 t_a = normalize(target_clamp - bone_root);

    quat r2 = quat_from_angle_axis(
        acosf(clampf(dot(c_a, t_a), -1.0f, 1.0f)),
        normalize(cross(c_a, t_a)));
    
    bone_root_lr = quat_inv_mul(bone_par_gr, quat_mul(r2, quat_mul(r0, bone_root_gr)));
    bone_mid_lr = quat_inv_mul(bone_root_gr, quat_mul(r1, bone_mid_gr));
}

//--------------------------------------

void draw_axis(const vec3 pos, const quat rot, const float scale = 1.0f)
{
    vec3 axis0 = pos + quat_mul_vec3(rot, scale * vec3(1.0f, 0.0f, 0.0f));
    vec3 axis1 = pos + quat_mul_vec3(rot, scale * vec3(0.0f, 1.0f, 0.0f));
    vec3 axis2 = pos + quat_mul_vec3(rot, scale * vec3(0.0f, 0.0f, 1.0f));
    
    DrawLine3D(to_Vector3(pos), to_Vector3(axis0), RED);
    DrawLine3D(to_Vector3(pos), to_Vector3(axis1), GREEN);
    DrawLine3D(to_Vector3(pos), to_Vector3(axis2), BLUE);
}

float bootstrap_support_root_offset(
    const slice1d<vec3> bone_positions,
    const slice1d<quat> bone_rotations,
    const slice1d<bool> contact_states,
    const slice1d<int> contact_bones,
    const slice1d<int> bone_parents,
    const slice1d<environment_box> environment_boxes,
    const float foot_height)
{
    array1d<vec3> global_positions(bone_positions.size); global_positions.zero();
    array1d<quat> global_rotations(bone_rotations.size); global_rotations.zero();
    array1d<bool> global_computed(bone_positions.size); global_computed.set(false);

    float offset_sum = 0.0f;
    int active_contacts = 0;

    for (int i = 0; i < contact_bones.size; i++)
    {
        if (!contact_states(i))
        {
            continue;
        }

        int toe_bone = contact_bones(i);
        forward_kinematics_partial(
            global_positions,
            global_rotations,
            global_computed,
            bone_positions,
            bone_rotations,
            bone_parents,
            toe_bone);

        float ground_height = environment_surface_height(
            global_positions(toe_bone).x,
            global_positions(toe_bone).z,
            environment_boxes);
        offset_sum += (ground_height + foot_height) - global_positions(toe_bone).y;
        active_contacts++;
    }

    return active_contacts > 0 ? offset_sum / active_contacts : 0.0f;
}

void draw_skeleton(
    const slice1d<vec3> global_bone_positions,
    const slice1d<int> bone_parents,
    const Color joint_color = DARKGRAY,
    const Color line_color = BLACK,
    const vec3 offset = vec3())
{
    for (int i = 0; i < global_bone_positions.size; i++)
    {
        vec3 joint_position = global_bone_positions(i) + offset;
        DrawSphereWires(to_Vector3(joint_position), i == 0 ? 0.05f : 0.035f, 4, 8, joint_color);

        if (bone_parents(i) != -1)
        {
            DrawLine3D(
                to_Vector3(global_bone_positions(bone_parents(i)) + offset),
                to_Vector3(joint_position),
                line_color);
        }
    }

    draw_axis(global_bone_positions(0) + offset, quat(), 0.2f);
}

void draw_features(const slice1d<float> features, const vec3 pos, const quat rot, const Color color)
{
    vec3 lfoot_pos = quat_mul_vec3(rot, vec3(features( 0), features( 1), features( 2))) + pos;
    vec3 rfoot_pos = quat_mul_vec3(rot, vec3(features( 3), features( 4), features( 5))) + pos;
    vec3 lfoot_vel = quat_mul_vec3(rot, vec3(features( 6), features( 7), features( 8)));
    vec3 rfoot_vel = quat_mul_vec3(rot, vec3(features( 9), features(10), features(11)));
    //vec3 hip_vel   = quat_mul_vec3(rot, vec3(features(12), features(13), features(14)));
    vec3 traj0_pos = quat_mul_vec3(rot, vec3(features(15),         0.0f, features(16))) + pos;
    vec3 traj1_pos = quat_mul_vec3(rot, vec3(features(17),         0.0f, features(18))) + pos;
    vec3 traj2_pos = quat_mul_vec3(rot, vec3(features(19),         0.0f, features(20))) + pos;
    vec3 traj0_dir = quat_mul_vec3(rot, vec3(features(21),         0.0f, features(22)));
    vec3 traj1_dir = quat_mul_vec3(rot, vec3(features(23),         0.0f, features(24)));
    vec3 traj2_dir = quat_mul_vec3(rot, vec3(features(25),         0.0f, features(26)));
    
    DrawSphereWires(to_Vector3(lfoot_pos), 0.05f, 4, 10, color);
    DrawSphereWires(to_Vector3(rfoot_pos), 0.05f, 4, 10, color);
    DrawSphereWires(to_Vector3(traj0_pos), 0.05f, 4, 10, color);
    DrawSphereWires(to_Vector3(traj1_pos), 0.05f, 4, 10, color);
    DrawSphereWires(to_Vector3(traj2_pos), 0.05f, 4, 10, color);
    
    DrawLine3D(to_Vector3(lfoot_pos), to_Vector3(lfoot_pos + 0.1f * lfoot_vel), color);
    DrawLine3D(to_Vector3(rfoot_pos), to_Vector3(rfoot_pos + 0.1f * rfoot_vel), color);
    
    DrawLine3D(to_Vector3(traj0_pos), to_Vector3(traj0_pos + 0.25f * traj0_dir), color);
    DrawLine3D(to_Vector3(traj1_pos), to_Vector3(traj1_pos + 0.25f * traj1_dir), color);
    DrawLine3D(to_Vector3(traj2_pos), to_Vector3(traj2_pos + 0.25f * traj2_dir), color); 

    if (features.size >= MOTION_MATCH_BASE_FEATURE_COUNT + terrain_feature_count())
    {
        float root_ground_height = environment_base_height(pos.x, pos.z);
        vec3 up = vec3(0.0f, 1.0f, 0.0f);

        auto draw_terrain_strip = [&](vec3 traj_pos, vec3 traj_dir, int terrain_offset)
        {
            vec3 dir = normalize(vec3(traj_dir.x, 0.0f, traj_dir.z) + vec3(1e-5f, 0.0f, 0.0f));
            vec3 right = normalize(cross(up, dir));

            vec3 sample_right = traj_pos + g_terrain_sampling_config.strip_half_width * right;
            vec3 sample_center = traj_pos;
            vec3 sample_left = traj_pos - g_terrain_sampling_config.strip_half_width * right;

            sample_right.y = root_ground_height + features(terrain_offset + 0);
            sample_center.y = root_ground_height + features(terrain_offset + 1);
            sample_left.y = root_ground_height + features(terrain_offset + 2);

            DrawSphere(to_Vector3(sample_right), 0.035f, DARKGREEN);
            DrawSphere(to_Vector3(sample_center), 0.035f, GREEN);
            DrawSphere(to_Vector3(sample_left), 0.035f, DARKGREEN);
            DrawLine3D(to_Vector3(sample_left), to_Vector3(sample_center), DARKGREEN);
            DrawLine3D(to_Vector3(sample_center), to_Vector3(sample_right), DARKGREEN);
        };

        int terrain_offset0 = environment_feature_offset_for_horizon(g_terrain_sampling_config.matching_trajectory_horizons[0]);
        int terrain_offset1 = environment_feature_offset_for_horizon(g_terrain_sampling_config.matching_trajectory_horizons[1]);
        int terrain_offset2 = environment_feature_offset_for_horizon(g_terrain_sampling_config.matching_trajectory_horizons[2]);
        if (terrain_offset0 != -1) draw_terrain_strip(traj0_pos, traj0_dir, MOTION_MATCH_BASE_FEATURE_COUNT + terrain_offset0);
        if (terrain_offset1 != -1) draw_terrain_strip(traj1_pos, traj1_dir, MOTION_MATCH_BASE_FEATURE_COUNT + terrain_offset1);
        if (terrain_offset2 != -1) draw_terrain_strip(traj2_pos, traj2_dir, MOTION_MATCH_BASE_FEATURE_COUNT + terrain_offset2);
    }

    if (features.size >= MOTION_MATCH_BASE_FEATURE_COUNT + terrain_feature_count() + obstacle_sdf_feature_count())
    {
        vec3 up = vec3(0.0f, 1.0f, 0.0f);

        auto draw_sdf_strip = [&](vec3 traj_pos, vec3 traj_dir, int sdf_offset)
        {
            vec3 dir = normalize(vec3(traj_dir.x, 0.0f, traj_dir.z) + vec3(1e-5f, 0.0f, 0.0f));
            vec3 right = normalize(cross(up, dir));

            vec3 samples[3] = {
                traj_pos + g_terrain_sampling_config.strip_half_width * right,
                traj_pos,
                traj_pos - g_terrain_sampling_config.strip_half_width * right,
            };

            for (int i = 0; i < 3; i++)
            {
                float sdf = features(sdf_offset + i);
                float danger = sdf < 0.0f ? 1.0f : 1.0f - clampf(sdf / g_terrain_sampling_config.sdf_clamp_distance, 0.0f, 1.0f);
                float radius = 0.05f + 0.12f * danger;

                samples[i].y = environment_base_height(samples[i].x, samples[i].z) + 0.03f;

                Color sdf_color = sdf < 0.0f ? MAROON :
                    (sdf < 0.25f ? RED :
                    (sdf < 0.75f ? ORANGE : GRAY));

                DrawSphereWires(to_Vector3(samples[i]), radius, 5, 8, sdf_color);
            }
        };

        int terrain_offset0 = environment_feature_offset_for_horizon(g_terrain_sampling_config.matching_trajectory_horizons[0]);
        int terrain_offset1 = environment_feature_offset_for_horizon(g_terrain_sampling_config.matching_trajectory_horizons[1]);
        int terrain_offset2 = environment_feature_offset_for_horizon(g_terrain_sampling_config.matching_trajectory_horizons[2]);
        if (terrain_offset0 != -1) draw_sdf_strip(traj0_pos, traj0_dir, MOTION_MATCH_BASE_FEATURE_COUNT + terrain_feature_count() + terrain_offset0);
        if (terrain_offset1 != -1) draw_sdf_strip(traj1_pos, traj1_dir, MOTION_MATCH_BASE_FEATURE_COUNT + terrain_feature_count() + terrain_offset1);
        if (terrain_offset2 != -1) draw_sdf_strip(traj2_pos, traj2_dir, MOTION_MATCH_BASE_FEATURE_COUNT + terrain_feature_count() + terrain_offset2);
    }
}

void draw_trajectory(
    const slice1d<vec3> trajectory_positions, 
    const slice1d<quat> trajectory_rotations, 
    const Color color)
{
    for (int i = 1; i < trajectory_positions.size; i++)
    {
        DrawSphereWires(to_Vector3(trajectory_positions(i)), 0.05f, 4, 10, color);
        DrawLine3D(to_Vector3(trajectory_positions(i)), to_Vector3(
            trajectory_positions(i) + 0.6f * quat_mul_vec3(trajectory_rotations(i), vec3(0, 0, 1.0f))), color);
        DrawLine3D(to_Vector3(trajectory_positions(i-1)), to_Vector3(trajectory_positions(i)), color);
    }
}

void draw_environment_boxes(
    const slice1d<environment_box> environment_boxes)
{
    for (int i = 0; i < environment_boxes.size; i++)
    {
        const environment_box& box = environment_boxes(i);

        vec3 position = vec3(
            box.position.x,
            environment_box_base(box) + 0.5f * box.scale.y + 0.01f,
            box.position.z);
      
        DrawCube(
            to_Vector3(position),
            box.scale.x,
            box.scale.y,
            box.scale.z,
            box.fill);
            
        DrawCubeWires(
            to_Vector3(position),
            box.scale.x,
            box.scale.y,
            box.scale.z,
            box.wire);
    }
}

//--------------------------------------

vec3 adjust_character_position(
    const vec3 character_position,
    const vec3 simulation_position,
    const float halflife,
    const float dt)
{
    // Find the difference in positioning
    vec3 difference_position = simulation_position - character_position;
    
    // Damp that difference using the given halflife and dt
    vec3 adjustment_position = damp_adjustment_exact(
        difference_position,
        halflife,
        dt);
    
    // Add the damped difference to move the character toward the sim
    return adjustment_position + character_position;
}

quat adjust_character_rotation(
    const quat character_rotation,
    const quat simulation_rotation,
    const float halflife,
    const float dt)
{
    // Find the difference in rotation (from character to simulation).
    // Here `quat_abs` forces the quaternion to take the shortest 
    // path and normalization is required as sometimes taking 
    // the difference between two very similar rotations can 
    // introduce numerical instability
    quat difference_rotation = quat_abs(quat_normalize(
        quat_mul_inv(simulation_rotation, character_rotation)));
    
    // Damp that difference using the given halflife and dt
    quat adjustment_rotation = damp_adjustment_exact(
        difference_rotation,
        halflife,
        dt);
    
    // Apply the damped adjustment to the character
    return quat_mul(adjustment_rotation, character_rotation);
}

vec3 adjust_character_position_by_velocity(
    const vec3 character_position,
    const vec3 character_velocity,
    const vec3 simulation_position,
    const float max_adjustment_ratio,
    const float halflife,
    const float dt)
{
    // Find and damp the desired adjustment
    vec3 adjustment_position = damp_adjustment_exact(
        simulation_position - character_position,
        halflife,
        dt);
    
    // If the length of the adjustment is greater than the character velocity 
    // multiplied by the ratio then we need to clamp it to that length
    float max_length = max_adjustment_ratio * length(character_velocity) * dt;
    
    if (length(adjustment_position) > max_length)
    {
        adjustment_position = max_length * normalize(adjustment_position);
    }
    
    // Apply the adjustment
    return adjustment_position + character_position;
}

quat adjust_character_rotation_by_velocity(
    const quat character_rotation,
    const vec3 character_angular_velocity,
    const quat simulation_rotation,
    const float max_adjustment_ratio,
    const float halflife,
    const float dt)
{
    // Find and damp the desired rotational adjustment
    quat adjustment_rotation = damp_adjustment_exact(
        quat_abs(quat_normalize(quat_mul_inv(
            simulation_rotation, character_rotation))),
        halflife,
        dt);
    
    // If the length of the adjustment is greater than the angular velocity 
    // multiplied by the ratio then we need to clamp this adjustment
    float max_length = max_adjustment_ratio *
        length(character_angular_velocity) * dt;
    
    if (length(quat_to_scaled_angle_axis(adjustment_rotation)) > max_length)
    {
        // To clamp can convert to scaled angle axis, rescale, and convert back
        adjustment_rotation = quat_from_scaled_angle_axis(max_length * 
            normalize(quat_to_scaled_angle_axis(adjustment_rotation)));
    }
    
    // Apply the adjustment
    return quat_mul(adjustment_rotation, character_rotation);
}

//--------------------------------------

vec3 clamp_character_position(
    const vec3 character_position,
    const vec3 simulation_position,
    const float max_distance)
{
    // If the character deviates too far from the simulation 
    // position we need to clamp it to within the max distance
    if (length(character_position - simulation_position) > max_distance)
    {
        return max_distance * 
            normalize(character_position - simulation_position) + 
            simulation_position;
    }
    else
    {
        return character_position;
    }
}
  
quat clamp_character_rotation(
    const quat character_rotation,
    const quat simulation_rotation,
    const float max_angle)
{
    // If the angle between the character rotation and simulation 
    // rotation exceeds the threshold we need to clamp it back
    if (quat_angle_between(character_rotation, simulation_rotation) > max_angle)
    {
        // First, find the rotational difference between the two
        quat diff = quat_abs(quat_mul_inv(
            character_rotation, simulation_rotation));
        
        // We can then decompose it into angle and axis
        float diff_angle; vec3 diff_axis;
        quat_to_angle_axis(diff, diff_angle, diff_axis);
        
        // We then clamp the angle to within our bounds
        diff_angle = clampf(diff_angle, -max_angle, max_angle);
        
        // And apply back the clamped rotation
        return quat_mul(
          quat_from_angle_axis(diff_angle, diff_axis), simulation_rotation);
    }
    else
    {
        return character_rotation;
    }
}

//--------------------------------------

void update_callback(void* args)
{
    ((std::function<void()>*)args)->operator()();
}

int main(int argc, char** argv)
{
    if (!eval_parse_args(argc, argv))
    {
        fprintf(stderr, "invalid eval arguments\n");
        return 1;
    }
    if (g_eval_config.selector_mode != "teacher" && g_eval_config.selector_mode != "learned")
    {
        g_eval_config.selector_mode = "learned";
    }
    if (g_eval_config.residual_mode != "teacher" && g_eval_config.residual_mode != "learned" && g_eval_config.residual_mode != "zero")
    {
        g_eval_config.residual_mode = "learned";
    }

    // Init Window
    
    const int screen_width = 1280;
    const int screen_height = 720;
    
    if (g_eval_config.enabled)
    {
        SetConfigFlags(FLAG_WINDOW_HIDDEN);
    }
    SetConfigFlags(FLAG_VSYNC_HINT);
    SetConfigFlags(FLAG_MSAA_4X_HINT);
    InitWindow(screen_width, screen_height, "raylib [data vs code driven displacement]");
    SetTargetFPS(60);
    
    // Camera

    Camera3D camera = { 0 };
    camera.position = (Vector3){ 0.0f, 10.0f, 10.0f };
    camera.target = (Vector3){ 0.0f, 0.0f, 0.0f };
    camera.up = (Vector3){ 0.0f, 1.0f, 0.0f };
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    float camera_azimuth = 0.0f;
    float camera_altitude = 0.4f;
    float camera_distance = 4.0f;
    bool terrain_preview_enabled = false;
    bool debug_draw_mesh = true;
    bool debug_draw_skeleton = true;
    bool debug_draw_bind_skeleton = false;
    bool debug_draw_bind_mesh = false;

    load_terrain_sampling_config(PFNN_TERRAIN_CONFIG_FILENAME);
    load_environment_heightmap();
    
    // Scene Obstacles
    
    array1d<environment_box> environment_boxes(0);
    initialize_environment_boxes(environment_boxes);

    // Ground Plane

    Shader ground_plane_shader = { 0 };
    float terrain_span_x = g_environment_heightmap.loaded() ? (g_environment_heightmap.x_max - g_environment_heightmap.x_min) : 28.0f;
    float terrain_span_z = g_environment_heightmap.loaded() ? (g_environment_heightmap.z_max - g_environment_heightmap.z_min) : 28.0f;
    int terrain_slices_x = g_environment_heightmap.loaded() ? (g_environment_heightmap.width > 2 ? g_environment_heightmap.width - 1 : 2) : 80;
    int terrain_slices_z = g_environment_heightmap.loaded() ? (g_environment_heightmap.height > 2 ? g_environment_heightmap.height - 1 : 2) : 80;
    float terrain_center_x = g_environment_heightmap.loaded() ? 0.5f * (g_environment_heightmap.x_min + g_environment_heightmap.x_max) : 0.0f;
    float terrain_center_z = g_environment_heightmap.loaded() ? 0.5f * (g_environment_heightmap.z_min + g_environment_heightmap.z_max) : 0.0f;

    Mesh ground_plane_mesh = { 0 };
    Model ground_plane_model = { 0 };

    auto rebuild_environment_scene = [&]()
    {
        if (!g_eval_config.enabled)
        {
            terrain_mesh_apply_heightfield(ground_plane_mesh, true);
        }
    };

    if (!g_eval_config.enabled)
    {
        ground_plane_shader = LoadShader(
            CHECKERBOARD_VERTEX_SHADER_FILENAME,
            CHECKERBOARD_FRAGMENT_SHADER_FILENAME);
        ground_plane_mesh = GenMeshPlane(terrain_span_x, terrain_span_z, terrain_slices_x, terrain_slices_z);
        terrain_mesh_offset_xz(ground_plane_mesh, terrain_center_x, terrain_center_z);
        terrain_mesh_apply_heightfield(ground_plane_mesh);
        ground_plane_model = LoadModelFromMesh(ground_plane_mesh);
        ground_plane_model.materials[0].shader = ground_plane_shader;
        rebuild_environment_scene();
    }
    
    // Character
    
    character character_data;
    Shader character_shader = { 0 };
    Mesh character_mesh = { 0 };
    Model character_model = { 0 };
    if (!g_eval_config.enabled)
    {
        character_load(character_data, "./resources/character.bin");
        character_shader = LoadShader(
            CHARACTER_VERTEX_SHADER_FILENAME,
            CHARACTER_FRAGMENT_SHADER_FILENAME);
        character_mesh = make_character_mesh(character_data);
        character_model = LoadModelFromMesh(character_mesh);
        character_model.materials[0].shader = character_shader;
    }
    
    // Load Animation Data and build Matching Database
    
    database db;
    database_load(db, "./resources/database.bin");
    
    float feature_weight_foot_position = 0.75f;
    float feature_weight_foot_velocity = 1.0f;
    float feature_weight_hip_velocity = 1.0f;
    float feature_weight_trajectory_positions = 1.0f;
    float feature_weight_trajectory_directions = 1.5f;
    float feature_weight_environment = 1.0f;

    auto rebuild_matching_database = [&]()
    {
        database_build_matching_features(
            db,
            feature_weight_foot_position,
            feature_weight_foot_velocity,
            feature_weight_hip_velocity,
            feature_weight_trajectory_positions,
            feature_weight_trajectory_directions,
            feature_weight_environment,
            PFNN_ENVIRONMENT_FEATURES_FILENAME);

        database_save_matching_features(db, "./resources/features.bin");
    };

    rebuild_matching_database();
   
    // Pose & Inertializer Data
    
    int frame_index = database_find_first_searchable_frame(db, 0);
    float inertialize_blending_halflife = 0.1f;

    array1d<vec3> curr_bone_positions = db.bone_positions(frame_index);
    array1d<vec3> curr_bone_velocities = db.bone_velocities(frame_index);
    array1d<quat> curr_bone_rotations = db.bone_rotations(frame_index);
    array1d<vec3> curr_bone_angular_velocities = db.bone_angular_velocities(frame_index);
    array1d<bool> curr_bone_contacts = db.contact_states(frame_index);

    array1d<vec3> trns_bone_positions = db.bone_positions(frame_index);
    array1d<vec3> trns_bone_velocities = db.bone_velocities(frame_index);
    array1d<quat> trns_bone_rotations = db.bone_rotations(frame_index);
    array1d<vec3> trns_bone_angular_velocities = db.bone_angular_velocities(frame_index);
    array1d<bool> trns_bone_contacts = db.contact_states(frame_index);

    array1d<vec3> bone_positions = db.bone_positions(frame_index);
    array1d<vec3> bone_velocities = db.bone_velocities(frame_index);
    array1d<quat> bone_rotations = db.bone_rotations(frame_index);
    array1d<vec3> bone_angular_velocities = db.bone_angular_velocities(frame_index);
    
    array1d<vec3> bone_offset_positions(db.nbones());
    array1d<vec3> bone_offset_velocities(db.nbones());
    array1d<quat> bone_offset_rotations(db.nbones());
    array1d<vec3> bone_offset_angular_velocities(db.nbones());
    
    array1d<vec3> global_bone_positions(db.nbones());
    array1d<vec3> global_bone_velocities(db.nbones());
    array1d<quat> global_bone_rotations(db.nbones());
    array1d<vec3> global_bone_angular_velocities(db.nbones());
    array1d<bool> global_bone_computed(db.nbones());
    
    vec3 transition_src_position;
    quat transition_src_rotation;
    vec3 transition_dst_position;
    quat transition_dst_rotation;
    
    inertialize_pose_reset(
        bone_offset_positions,
        bone_offset_velocities,
        bone_offset_rotations,
        bone_offset_angular_velocities,
        transition_src_position,
        transition_src_rotation,
        transition_dst_position,
        transition_dst_rotation,
        bone_positions(0),
        bone_rotations(0));
    
    inertialize_pose_update(
        bone_positions,
        bone_velocities,
        bone_rotations,
        bone_angular_velocities,
        bone_offset_positions,
        bone_offset_velocities,
        bone_offset_rotations,
        bone_offset_angular_velocities,
        db.bone_positions(frame_index),
        db.bone_velocities(frame_index),
        db.bone_rotations(frame_index),
        db.bone_angular_velocities(frame_index),
        transition_src_position,
        transition_src_rotation,
        transition_dst_position,
        transition_dst_rotation,
        inertialize_blending_halflife,
        0.0f);
        
    // Trajectory & Gameplay Data
    
    float search_time = 0.1f;
    float search_timer = search_time;
    float force_search_timer = search_time;
    float motion_matching_transition_cost = 0.10f;
    
    vec3 desired_velocity;
    vec3 desired_velocity_change_curr;
    vec3 desired_velocity_change_prev;
    float desired_velocity_change_threshold = 50.0;
    
    quat desired_rotation;
    vec3 desired_rotation_change_curr;
    vec3 desired_rotation_change_prev;
    float desired_rotation_change_threshold = 50.0;
    
    float desired_gait = 0.0f;
    float desired_gait_velocity = 0.0f;
    
    vec3 simulation_position;
    vec3 simulation_velocity;
    vec3 simulation_acceleration;
    quat simulation_rotation;
    vec3 simulation_angular_velocity;
    
    float simulation_velocity_halflife = 0.27f;
    float simulation_rotation_halflife = 0.27f;
    
    // Query/request speeds in m/s. When enabled below, root motion follows
    // the matched animation speed instead of these values directly.
    float simulation_run_fwrd_speed = 4.0f;
    float simulation_run_side_speed = 3.0f;
    float simulation_run_back_speed = 2.5f;
    
    float simulation_walk_fwrd_speed = 1.75f;
    float simulation_walk_side_speed = 1.5f;
    float simulation_walk_back_speed = 1.25f;
    bool simulation_follow_animation_speed = true;
    float lmm_projector_transition_threshold = 0.10f;
    float lmm_projector_blend_halflife = 0.06f;
    float trajectory_prediction_dt = g_terrain_sampling_config.prediction_frame_step * (1.0f / 60.0f);
    
    array1d<vec3> trajectory_desired_velocities(prediction_sample_count());
    array1d<quat> trajectory_desired_rotations(prediction_sample_count());
    array1d<vec3> trajectory_positions(prediction_sample_count());
    array1d<vec3> trajectory_velocities(prediction_sample_count());
    array1d<vec3> trajectory_accelerations(prediction_sample_count());
    array1d<quat> trajectory_rotations(prediction_sample_count());
    array1d<vec3> trajectory_angular_velocities(prediction_sample_count());
    
    // Synchronization
    
    bool synchronization_enabled = false;
    float synchronization_data_factor = 1.0f;
    
    // Adjustment
    
    bool adjustment_enabled = true;
    bool adjustment_by_velocity_enabled = true;
    float adjustment_position_halflife = 0.1f;
    float adjustment_rotation_halflife = 0.2f;
    float adjustment_position_max_ratio = 0.5f;
    float adjustment_rotation_max_ratio = 0.5f;
    
    // Clamping
    
    bool clamping_enabled = true;
    float clamping_max_distance = 0.15f;
    float clamping_max_angle = 0.5f * PIf;
    
    // IK
    
    bool ik_enabled = true;
    float ik_max_length_buffer = 0.015f;
    float ik_foot_height = 0.02f;
    float ik_toe_length = 0.15f;
    float ik_unlock_radius = 0.2f;
    float ik_blending_halflife = 0.1f;
    
    // Contact and Foot Locking data
    
    array1d<int> contact_bones(2);
    contact_bones(0) = Bone_LeftToe;
    contact_bones(1) = Bone_RightToe;
    
    array1d<bool> contact_states(contact_bones.size);
    array1d<bool> contact_locks(contact_bones.size);
    array1d<vec3> contact_positions(contact_bones.size);
    array1d<vec3> contact_velocities(contact_bones.size);
    array1d<vec3> contact_points(contact_bones.size);
    array1d<vec3> contact_targets(contact_bones.size);
    array1d<vec3> contact_offset_positions(contact_bones.size);
    array1d<vec3> contact_offset_velocities(contact_bones.size);
    
    for (int i = 0; i < contact_bones.size; i++)
    {
        vec3 bone_position;
        vec3 bone_velocity;
        quat bone_rotation;
        vec3 bone_angular_velocity;
        
        forward_kinematics_velocity(
            bone_position,
            bone_velocity,
            bone_rotation,
            bone_angular_velocity,
            bone_positions,
            bone_velocities,
            bone_rotations,
            bone_angular_velocities,
            db.bone_parents,
            contact_bones(i));
        
        contact_reset(
            contact_states(i),
            contact_locks(i),
            contact_positions(i),  
            contact_velocities(i),
            contact_points(i),
            contact_targets(i),
            contact_offset_positions(i),
            contact_offset_velocities(i),
            bone_position,
            bone_velocity,
            false);
    }
    
    array1d<vec3> adjusted_bone_positions = bone_positions;
    array1d<quat> adjusted_bone_rotations = bone_rotations;
    
    // Learned Motion Matching
    
    bool lmm_enabled = false;
    const int hybrid_topk = 8;

    nnet decompressor, selector, residual_stepper;
    array2d<float> latent_database;
    bool hybrid_assets_available =
        file_exists("./resources/decompressor.bin") &&
        file_exists("./resources/selector.bin") &&
        file_exists("./resources/residual_stepper.bin") &&
        file_exists("./resources/latent.bin");

    bool lmm_networks_compatible = false;
    nnet_evaluation decompressor_evaluation, selector_evaluation, residual_stepper_evaluation;

    if (hybrid_assets_available)
    {
        nnet_load(decompressor, "./resources/decompressor.bin");
        nnet_load(selector, "./resources/selector.bin");
        nnet_load(residual_stepper, "./resources/residual_stepper.bin");
        latent_load(latent_database, "./resources/latent.bin");

        decompressor_evaluation.resize(decompressor);
        selector_evaluation.resize(selector);
        residual_stepper_evaluation.resize(residual_stepper);

        lmm_networks_compatible =
            decompressor.input_mean.size == db.nfeatures() + 32 &&
            selector.input_mean.size == db.nfeatures() * 3 + 1 &&
            residual_stepper.input_mean.size == db.nfeatures() + 32 + control_feature_count(db.nfeatures()) + db.nfeatures() + 32 &&
            residual_stepper.output_mean.size == db.nfeatures() + 32 &&
            latent_database.rows == db.nframes() &&
            latent_database.cols == 32;
    }

    lmm_enabled = lmm_networks_compatible;

    array1d<float> features_curr = db.features(frame_index);
    array1d<float> latent_curr(32); latent_curr.zero();
    array1d<float> residual_features(db.nfeatures()); residual_features.zero();
    array1d<float> residual_latent(32); residual_latent.zero();
    array1d<int> topk_indices(hybrid_topk); topk_indices.set(-1);
    array1d<float> topk_costs(hybrid_topk); topk_costs.set(FLT_MAX);
    array1d<float> selector_scores(hybrid_topk); selector_scores.set(-FLT_MAX);
    int exemplar_index = -1;
    array1d<float> stepper_control(control_feature_count(db.nfeatures())); stepper_control.zero();
    array1d<float> debug_query_features = db.features(frame_index);
    denormalize_features(debug_query_features, db.features_offset, db.features_scale);

    if (lmm_networks_compatible)
    {
        for (int i = 0; i < latent_curr.size; i++)
        {
            latent_curr(i) = latent_database(frame_index, i);
        }
    }

    FILE* eval_trace = NULL;
    int eval_frame_index = 0;
    bool eval_complete = false;
    int teacher_frame_index = frame_index;
    bool debug_teacher_selector = g_eval_config.selector_mode == "teacher";
    bool debug_learned_selector = g_eval_config.selector_mode == "learned";
    bool debug_teacher_residual = g_eval_config.residual_mode == "teacher";
    bool debug_learned_residual = g_eval_config.residual_mode == "learned";
    bool debug_zero_residual = g_eval_config.residual_mode == "zero";
    std::vector<runtime_debug_snapshot> failure_history;
    std::vector<runtime_debug_snapshot> failure_future;
    bool failure_triggered = false;
    bool failure_dump_written = false;
    int failure_future_remaining = 0;
    array1d<vec3> eval_prev_contact_positions(contact_bones.size);
    array1d<bool> eval_prev_contact_valid(contact_bones.size);
    eval_prev_contact_positions.zero();
    eval_prev_contact_valid.set(false);
    int exemplar_dwell_frames = 1000000;
    float runtime_prev_contact_slip = 0.0f;
    float runtime_prev_contact_height_error = 0.0f;
    int runtime_prev_terrain_penetrations = 0;
    bool runtime_emergency_reanchor = false;
    
    // Go

    float dt = 1.0f / 60.0f;

    simulation_position = vec3(g_eval_config.start_x, environment_surface_height(g_eval_config.start_x, g_eval_config.start_z, environment_boxes), g_eval_config.start_z);
    simulation_velocity = vec3();
    simulation_acceleration = vec3();
    simulation_rotation = quat_from_angle_axis(g_eval_config.start_yaw, vec3(0.0f, 1.0f, 0.0f));
    simulation_angular_velocity = vec3();
    desired_velocity = vec3();
    desired_velocity_change_curr = vec3();
    desired_velocity_change_prev = vec3();
    desired_rotation = simulation_rotation;
    desired_rotation_change_curr = vec3();
    desired_rotation_change_prev = vec3();
    lmm_enabled = g_eval_config.enabled ? (g_eval_config.use_learned && lmm_networks_compatible) : lmm_enabled;
    if (g_eval_config.enabled)
    {
        terrain_preview_enabled = false;
        debug_draw_mesh = false;
        debug_draw_skeleton = false;
        debug_draw_bind_skeleton = false;
        debug_draw_bind_mesh = false;
        eval_trace = fopen(g_eval_config.trace_path.c_str(), "w");
        assert(eval_trace != NULL);
            fprintf(
                eval_trace,
                "frame,time,root_x,root_y,root_z,teacher_frame,selected_exemplar,exemplar_valid,switch_flag,divergence_flag,"
                "residual_norm_before,residual_norm_after,stepper_input_norm,stepper_output_norm,decompressor_input_norm,"
                "root_planar_speed,contact_height_error,contact_slip,terrain_penetrations,correction_magnitude,stepper_drift,"
                "switch_severity,residual_carry_alpha,selector_margin,exemplar_dwell_frames,emergency_reanchor,hard_reset,"
                "finite_query,finite_features,finite_latent,finite_residual,finite_stepper,finite_decompressor,finite_pose,mode,selector_mode,residual_mode\n");
    }

    auto update_func = [&]()
    {
        if (IsKeyPressed(KEY_T))
        {
            terrain_preview_enabled = !terrain_preview_enabled;
        }
        if (IsKeyPressed(KEY_B))
        {
            debug_draw_skeleton = !debug_draw_skeleton;
        }
        if (IsKeyPressed(KEY_M))
        {
            debug_draw_mesh = !debug_draw_mesh;
        }
        if (IsKeyPressed(KEY_N))
        {
            debug_draw_bind_skeleton = !debug_draw_bind_skeleton;
        }
        if (IsKeyPressed(KEY_R))
        {
            debug_draw_bind_mesh = !debug_draw_bind_mesh;
        }
        if (IsKeyPressed(KEY_I))
        {
            ik_enabled = !ik_enabled;
        }
      
        eval_input_sample eval_input;
        if (g_eval_config.enabled)
        {
            eval_input = g_eval_script[std::min(eval_frame_index, (int)g_eval_script.size() - 1)];
        }

        // Get gamepad stick states
        vec3 gamepadstick_left = g_eval_config.enabled ? vec3(eval_input.left_x, 0.0f, eval_input.left_z) : gamepad_get_stick(GAMEPAD_STICK_LEFT);
        vec3 gamepadstick_right = g_eval_config.enabled ? vec3(eval_input.right_x, 0.0f, eval_input.right_z) : gamepad_get_stick(GAMEPAD_STICK_RIGHT);
        
        // Get if strafe is desired
        bool desired_strafe = g_eval_config.enabled ? eval_input.strafe : desired_strafe_update();
        
        // Get the desired gait (walk / run)
        if (g_eval_config.enabled)
        {
            desired_gait = eval_input.gait;
            desired_gait_velocity = 0.0f;
        }
        else
        {
            desired_gait_update(
                desired_gait,
                desired_gait_velocity,
                dt);
        }
        
        // Get the desired query/request speeds based on the gait
        float simulation_fwrd_speed = 0.0f;
        float simulation_side_speed = 0.0f;
        float simulation_back_speed = 0.0f;
        locomotion_request_speeds_update(
            simulation_fwrd_speed,
            simulation_side_speed,
            simulation_back_speed,
            desired_gait,
            simulation_follow_animation_speed,
            simulation_run_fwrd_speed,
            simulation_run_side_speed,
            simulation_run_back_speed,
            simulation_walk_fwrd_speed,
            simulation_walk_side_speed,
            simulation_walk_back_speed);
        
        // Get the desired velocity
        vec3 desired_velocity_curr = desired_velocity_update(
            gamepadstick_left,
            camera_azimuth,
            simulation_rotation,
            simulation_fwrd_speed,
            simulation_side_speed,
            simulation_back_speed);
            
        // Get the desired rotation/direction
        quat desired_rotation_curr = desired_rotation_update(
            desired_rotation,
            gamepadstick_left,
            gamepadstick_right,
            camera_azimuth,
            desired_strafe,
            desired_velocity_curr);
        
        // Check if we should force a search because input changed quickly
        desired_velocity_change_prev = desired_velocity_change_curr;
        desired_velocity_change_curr =  (desired_velocity_curr - desired_velocity) / dt;
        desired_velocity = desired_velocity_curr;
        
        desired_rotation_change_prev = desired_rotation_change_curr;
        desired_rotation_change_curr = quat_to_scaled_angle_axis(quat_abs(quat_mul_inv(desired_rotation_curr, desired_rotation))) / dt;
        desired_rotation =  desired_rotation_curr;
        
        bool force_search = false;

        if (force_search_timer <= 0.0f && (
            (length(desired_velocity_change_prev) >= desired_velocity_change_threshold && 
             length(desired_velocity_change_curr)  < desired_velocity_change_threshold)
        ||  (length(desired_rotation_change_prev) >= desired_rotation_change_threshold && 
             length(desired_rotation_change_curr)  < desired_rotation_change_threshold)))
        {
            force_search = true;
            force_search_timer = search_time;
        }
        else if (force_search_timer > 0)
        {
            force_search_timer -= dt;
        }
        
        // Predict Future Trajectory
        
        trajectory_desired_rotations_predict(
          trajectory_desired_rotations,
          trajectory_desired_velocities,
          desired_rotation,
          camera_azimuth,
          gamepadstick_left,
          gamepadstick_right,
          desired_strafe,
          trajectory_prediction_dt);
        
        trajectory_rotations_predict(
            trajectory_rotations,
            trajectory_angular_velocities,
            simulation_rotation,
            simulation_angular_velocity,
            trajectory_desired_rotations,
            simulation_rotation_halflife,
            trajectory_prediction_dt);
        
        trajectory_desired_velocities_predict(
          trajectory_desired_velocities,
          trajectory_rotations,
          desired_velocity,
          camera_azimuth,
          gamepadstick_left,
          gamepadstick_right,
          desired_strafe,
          simulation_fwrd_speed,
          simulation_side_speed,
          simulation_back_speed,
          trajectory_prediction_dt);
        
        trajectory_positions_predict(
            trajectory_positions,
            trajectory_velocities,
            trajectory_accelerations,
            simulation_position,
            simulation_velocity,
            simulation_acceleration,
            trajectory_desired_velocities,
            simulation_velocity_halflife,
            trajectory_prediction_dt,
            environment_boxes);
           
        // Make query vector for search.
        // In theory this only needs to be done when a search is 
        // actually required however for visualization purposes it
        // can be nice to do it every frame
        array1d<float> query(db.nfeatures());
                
        // Compute the features of the query vector

        slice1d<float> query_features = lmm_enabled ? slice1d<float>(features_curr) : db.features(frame_index);

        int offset = 0;
        query_copy_denormalized_feature(query, offset, 3, query_features, db.features_offset, db.features_scale); // Left Foot Position
        query_copy_denormalized_feature(query, offset, 3, query_features, db.features_offset, db.features_scale); // Right Foot Position
        query_copy_denormalized_feature(query, offset, 3, query_features, db.features_offset, db.features_scale); // Left Foot Velocity
        query_copy_denormalized_feature(query, offset, 3, query_features, db.features_offset, db.features_scale); // Right Foot Velocity
        query_copy_denormalized_feature(query, offset, 3, query_features, db.features_offset, db.features_scale); // Hip Velocity
        query_compute_trajectory_position_feature(query, offset, bone_positions(0), bone_rotations(0), trajectory_positions);
        query_compute_trajectory_direction_feature(query, offset, bone_rotations(0), trajectory_rotations);
        query_copy_denormalized_feature(
            query,
            offset,
            MOTION_MATCH_PHASE_CONTACT_FEATURE_COUNT,
            query_features,
            db.features_offset,
            db.features_scale); // Phase sin/cos + contact state
        if (offset < db.nfeatures())
        {
            query_compute_environment_feature(
                query,
                offset,
                bone_positions(0),
                trajectory_positions,
                trajectory_rotations,
                environment_boxes,
                db.nfeatures() - offset);
        }
        
        assert(offset == db.nfeatures());
        debug_query_features = query;
        stepper_control.zero();

        array1d<float> query_normalized(db.nfeatures());
        for (int i = 0; i < db.nfeatures(); i++)
        {
            float feature_scale = fabsf(db.features_scale(i)) > 1e-8f ? db.features_scale(i) : 1.0f;
            query_normalized(i) = (query(i) - db.features_offset(i)) / feature_scale;
        }
        float projector_correction_mag = 0.0f;
        float stepper_drift_mag = 0.0f;
        float residual_norm_before = 0.0f;
        float residual_norm_after = 0.0f;
        float stepper_input_norm = 0.0f;
        float stepper_output_norm = 0.0f;
        float decompressor_input_norm = 0.0f;
        bool switch_flag = false;
        float switch_severity = 0.0f;
        float residual_carry_alpha = 1.0f;
        float selector_margin = 0.0f;
        bool emergency_reanchor = runtime_emergency_reanchor;
        bool hard_reset = false;
        bool exemplar_valid = exemplar_index >= 0 && exemplar_index < db.nframes() &&
            (db.searchable_frames.size != db.nframes() || db.searchable_frames(exemplar_index));

        // Check if we reached the end of the current anim
        bool end_of_anim = database_advance_frame(db, frame_index) == frame_index;
        
        bool lmm_reproject_enabled = lmm_enabled && lmm_networks_compatible;

        if (lmm_reproject_enabled)
        {
            search_timer -= dt;
            int teacher_next_frame = teacher_frame_index;
            int exemplar_next_frame = exemplar_index;
            bool teacher_end_of_anim = database_advance_frame(db, teacher_frame_index) == teacher_frame_index;
            bool exemplar_end_of_anim = true;
            if (exemplar_index >= 0)
            {
                exemplar_next_frame = database_advance_frame(db, exemplar_index);
                exemplar_end_of_anim = exemplar_next_frame == exemplar_index;
            }
            bool do_search =
                search_timer <= 0.0f ||
                force_search ||
                emergency_reanchor ||
                exemplar_index == -1 ||
                teacher_end_of_anim ||
                exemplar_end_of_anim;

            if (do_search)
            {
                database_search_topk(
                    topk_indices,
                    topk_costs,
                    db,
                    query,
                    exemplar_index,
                    motion_matching_transition_cost);

                selector_scores.set(-FLT_MAX);
                int selected_index = exemplar_index;
                teacher_next_frame = topk_indices(0) >= 0 ? topk_indices(0) : teacher_frame_index;
                float best_score = -FLT_MAX;
                float current_score = -FLT_MAX;
                bool current_in_topk = false;

                if (debug_teacher_selector)
                {
                    selected_index = teacher_next_frame;
                }
                else
                {
                    for (int k = 0; k < topk_indices.size; k++)
                    {
                        if (topk_indices(k) < 0)
                        {
                            continue;
                        }

                        float score = selector_score_candidate(
                            selector_evaluation,
                            query_normalized,
                            query_normalized,
                            db.features(topk_indices(k)),
                            topk_costs(k),
                            selector);
                        selector_scores(k) = score;

                        if (score > best_score)
                        {
                            best_score = score;
                            selected_index = topk_indices(k);
                        }

                        if (topk_indices(k) == exemplar_index)
                        {
                            current_score = score;
                            current_in_topk = true;
                        }
                    }
                }

                if (!debug_teacher_selector &&
                    selected_index != exemplar_index &&
                    exemplar_index >= 0 &&
                    exemplar_valid &&
                    current_in_topk)
                {
                    selector_margin = HYBRID_SELECTOR_SCORE_MARGIN;
                    bool dwell_locked = exemplar_dwell_frames < HYBRID_SELECTOR_MIN_DWELL_FRAMES;
                    if (!emergency_reanchor &&
                        best_score < current_score + selector_margin &&
                        dwell_locked)
                    {
                        selected_index = exemplar_index;
                    }
                }

                if (selected_index != -1)
                {
                    residual_norm_before = hybrid_residual_norm(residual_features, residual_latent);
                    for (int i = 0; i < features_curr.size; i++)
                    {
                        projector_correction_mag += squaref(db.features(selected_index, i) - features_curr(i));
                    }
                    projector_correction_mag = sqrtf(projector_correction_mag / maxf((float)features_curr.size, 1.0f));

                    switch_flag = selected_index != exemplar_index;
                    exemplar_index = selected_index;
                    exemplar_valid = db.searchable_frames.size != db.nframes() || db.searchable_frames(exemplar_index);
                    switch_severity = hybrid_switch_severity(
                        features_curr,
                        db.features(exemplar_index),
                        curr_bone_contacts,
                        db.contact_states(exemplar_index));

                    if (debug_teacher_residual)
                    {
                        teacher_frame_index = teacher_next_frame;
                        features_curr = db.features(teacher_frame_index);
                        for (int i = 0; i < latent_curr.size; i++)
                        {
                            latent_curr(i) = latent_database(teacher_frame_index, i);
                        }
                        for (int i = 0; i < residual_features.size; i++)
                        {
                            residual_features(i) = features_curr(i) - db.features(exemplar_index, i);
                        }
                        for (int i = 0; i < residual_latent.size; i++)
                        {
                            residual_latent(i) = latent_curr(i) - latent_database(exemplar_index, i);
                        }
                    }
                    else if (debug_zero_residual)
                    {
                        residual_features.zero();
                        residual_latent.zero();
                        features_curr = db.features(exemplar_index);
                        for (int i = 0; i < latent_curr.size; i++)
                        {
                            latent_curr(i) = latent_database(exemplar_index, i);
                        }
                        teacher_frame_index = teacher_next_frame;
                    }
                    else
                    {
                        teacher_frame_index = teacher_next_frame;
                        for (int i = 0; i < residual_features.size; i++)
                        {
                            residual_features(i) = features_curr(i) - db.features(exemplar_index, i);
                        }
                        for (int i = 0; i < residual_latent.size; i++)
                        {
                            residual_latent(i) = latent_curr(i) - latent_database(exemplar_index, i);
                        }

                        bool contact_mode_changed =
                            contact_state_mismatch(curr_bone_contacts, db.contact_states(exemplar_index)) > 0.5f;
                        if (emergency_reanchor ||
                            runtime_prev_contact_slip > HYBRID_CONTACT_SLIP_HARD_RESET ||
                            runtime_prev_terrain_penetrations > 0 ||
                            residual_norm_before > HYBRID_RESIDUAL_NORM_MAX * 0.9f ||
                            switch_severity >= HYBRID_SWITCH_HARD_SEVERITY ||
                            contact_mode_changed)
                        {
                            residual_carry_alpha = 0.0f;
                            hard_reset = true;
                        }
                        else if (switch_severity >= HYBRID_SWITCH_MEDIUM_SEVERITY ||
                                 runtime_prev_contact_slip > HYBRID_CONTACT_SLIP_GUARD ||
                                 residual_norm_before > HYBRID_RESIDUAL_NORM_P99 * 2.0f)
                        {
                            residual_carry_alpha = HYBRID_CARRY_ALPHA_MEDIUM;
                        }
                        else if (switch_severity >= HYBRID_SWITCH_SOFT_SEVERITY ||
                                 residual_norm_before > HYBRID_RESIDUAL_NORM_P99)
                        {
                            residual_carry_alpha = HYBRID_CARRY_ALPHA_SOFT;
                        }

                        for (int i = 0; i < residual_features.size; i++)
                        {
                            residual_features(i) *= residual_carry_alpha;
                        }
                        for (int i = 0; i < residual_latent.size; i++)
                        {
                            residual_latent(i) *= residual_carry_alpha;
                        }
                        hybrid_reconstruct_absolute_state(
                            features_curr,
                            latent_curr,
                            db.features(exemplar_index),
                            latent_database(exemplar_index),
                            residual_features,
                            residual_latent);
                    }

                    residual_norm_after = hybrid_residual_norm(residual_features, residual_latent);
                }

                runtime_emergency_reanchor = false;
                search_timer = search_time;
            }
            else if (exemplar_index >= 0 && !debug_teacher_residual)
            {
                exemplar_index = exemplar_next_frame;
                exemplar_valid =
                    db.searchable_frames.size != db.nframes() ||
                    db.searchable_frames(exemplar_index);

                if (debug_zero_residual)
                {
                    residual_features.zero();
                    residual_latent.zero();
                    features_curr = db.features(exemplar_index);
                    for (int i = 0; i < latent_curr.size; i++)
                    {
                        latent_curr(i) = latent_database(exemplar_index, i);
                    }
                }
                else
                {
                    for (int i = 0; i < residual_features.size; i++)
                    {
                        residual_features(i) = features_curr(i) - db.features(exemplar_index, i);
                    }
                    for (int i = 0; i < residual_latent.size; i++)
                    {
                        residual_latent(i) = latent_curr(i) - latent_database(exemplar_index, i);
                    }

                    residual_norm_before = hybrid_residual_norm(residual_features, residual_latent);
                    if (runtime_prev_contact_slip > HYBRID_CONTACT_SLIP_HARD_RESET ||
                        runtime_prev_terrain_penetrations > 0 ||
                        residual_norm_before > HYBRID_RESIDUAL_NORM_MAX * 0.9f)
                    {
                        residual_carry_alpha = 0.0f;
                        hard_reset = true;
                    }
                    else if (runtime_prev_contact_slip > HYBRID_CONTACT_SLIP_GUARD ||
                             residual_norm_before > HYBRID_RESIDUAL_NORM_P99 * 2.0f)
                    {
                        residual_carry_alpha = HYBRID_CARRY_ALPHA_MEDIUM;
                    }
                    else if (residual_norm_before > HYBRID_RESIDUAL_NORM_P99)
                    {
                        residual_carry_alpha = HYBRID_CARRY_ALPHA_SOFT;
                    }

                    for (int i = 0; i < residual_features.size; i++)
                    {
                        residual_features(i) *= residual_carry_alpha;
                    }
                    for (int i = 0; i < residual_latent.size; i++)
                    {
                        residual_latent(i) *= residual_carry_alpha;
                    }
                    hybrid_reconstruct_absolute_state(
                        features_curr,
                        latent_curr,
                        db.features(exemplar_index),
                        latent_database(exemplar_index),
                        residual_features,
                        residual_latent);
                    residual_norm_after = hybrid_residual_norm(residual_features, residual_latent);
                }
            }
            else if (debug_teacher_residual)
            {
                teacher_frame_index = database_advance_frame(db, teacher_frame_index);
                features_curr = db.features(teacher_frame_index);
                for (int i = 0; i < latent_curr.size; i++)
                {
                    latent_curr(i) = latent_database(teacher_frame_index, i);
                }
                for (int i = 0; i < residual_features.size; i++)
                {
                    residual_features(i) = features_curr(i) - db.features(exemplar_index, i);
                }
                for (int i = 0; i < residual_latent.size; i++)
                {
                    residual_latent(i) = latent_curr(i) - latent_database(exemplar_index, i);
                }
            }
        }
        else
        {
            // Search
            
            int best_index = end_of_anim ? -1 : frame_index;
            float best_cost = FLT_MAX;
            
            database_search(
                best_index,
                best_cost,
                db,
                query,
                motion_matching_transition_cost);
            
            // Transition if better frame found
            
            if (best_index != frame_index)
            {
                trns_bone_positions = db.bone_positions(best_index);
                trns_bone_velocities = db.bone_velocities(best_index);
                trns_bone_rotations = db.bone_rotations(best_index);
                trns_bone_angular_velocities = db.bone_angular_velocities(best_index);
                
                inertialize_pose_transition(
                    bone_offset_positions,
                    bone_offset_velocities,
                    bone_offset_rotations,
                    bone_offset_angular_velocities,
                    transition_src_position,
                    transition_src_rotation,
                    transition_dst_position,
                    transition_dst_rotation,
                    bone_positions(0),
                    bone_velocities(0),
                    bone_rotations(0),
                    bone_angular_velocities(0),
                    curr_bone_positions,
                    curr_bone_velocities,
                    curr_bone_rotations,
                    curr_bone_angular_velocities,
                    trns_bone_positions,
                    trns_bone_velocities,
                    trns_bone_rotations,
                    trns_bone_angular_velocities);
                
                frame_index = best_index;
            }

        }

        if (lmm_reproject_enabled)
        {
            exemplar_dwell_frames = switch_flag ? 0 : exemplar_dwell_frames + 1;
        }

        slice1d<float> drift_features = lmm_enabled && lmm_networks_compatible ? slice1d<float>(features_curr) : db.features(frame_index);
        for (int i = 0; i < drift_features.size; i++)
        {
            stepper_drift_mag += squaref(query_normalized(i) - drift_features(i));
        }
        stepper_drift_mag = sqrtf(stepper_drift_mag / maxf((float)drift_features.size, 1.0f));

        if (lmm_enabled && lmm_networks_compatible)
        {
            for (int i = 0; i < stepper_control.size; i++)
            {
                stepper_control(i) = features_curr(15 + i);
            }

            array1d<float> anchor_latent(32);
            for (int i = 0; i < anchor_latent.size; i++)
            {
                anchor_latent(i) = latent_database(exemplar_index, i);
            }

            if (debug_learned_residual)
            {
                array1d<float> residual_features_prev = residual_features;
                array1d<float> residual_latent_prev = residual_latent;
                stepper_input_norm = sqrtf(
                    squaref(array_norm(residual_features)) +
                    squaref(array_norm(residual_latent)) +
                    squaref(array_norm(stepper_control)) +
                    squaref(array_norm(db.features(exemplar_index))) +
                    squaref(array_norm(anchor_latent)));

                if (stepper_input_norm > HYBRID_STEPPER_INPUT_NORM_MAX ||
                    hybrid_residual_norm(residual_features, residual_latent) > HYBRID_RESIDUAL_NORM_MAX)
                {
                    residual_features.zero();
                    residual_latent.zero();
                    hard_reset = true;
                    emergency_reanchor = true;
                    runtime_emergency_reanchor = true;
                    hybrid_reconstruct_absolute_state(
                        features_curr,
                        latent_curr,
                        db.features(exemplar_index),
                        anchor_latent,
                        residual_features,
                        residual_latent);
                    stepper_input_norm = 0.0f;
                    stepper_output_norm = 0.0f;
                }
                else
                {
                    residual_stepper_evaluate(
                        residual_features,
                        residual_latent,
                        stepper_control,
                        db.features(exemplar_index),
                        anchor_latent,
                        residual_stepper_evaluation,
                        residual_stepper,
                        dt);

                    float diff_total = 0.0f;
                    for (int i = 0; i < residual_features.size; i++)
                    {
                        diff_total += squaref(residual_features(i) - residual_features_prev(i));
                    }
                    for (int i = 0; i < residual_latent.size; i++)
                    {
                        diff_total += squaref(residual_latent(i) - residual_latent_prev(i));
                    }
                    stepper_output_norm = sqrtf(diff_total);
                }
            }
            else if (debug_zero_residual)
            {
                stepper_input_norm = 0.0f;
                stepper_output_norm = 0.0f;
                residual_features.zero();
                residual_latent.zero();
            }

            for (int i = 0; i < features_curr.size; i++)
            {
                features_curr(i) = db.features(exemplar_index, i) + residual_features(i);
            }
            for (int i = 0; i < latent_curr.size; i++)
            {
                latent_curr(i) = latent_database(exemplar_index, i) + residual_latent(i);
            }
            decompressor_input_norm = sqrtf(squaref(array_norm(features_curr)) + squaref(array_norm(latent_curr)));
            
            // Decompress next pose
            decompressor_evaluate(
                curr_bone_positions,
                curr_bone_velocities,
                curr_bone_rotations,
                curr_bone_angular_velocities,
                curr_bone_contacts,
                decompressor_evaluation,
                features_curr,
                latent_curr,
                curr_bone_positions(0),
                curr_bone_rotations(0),
                decompressor,
                dt);
        }
        else
        {
            // Tick frame
            frame_index = database_advance_frame(db, frame_index);
            
            // Look-up Next Pose
            curr_bone_positions = db.bone_positions(frame_index);
            curr_bone_velocities = db.bone_velocities(frame_index);
            curr_bone_rotations = db.bone_rotations(frame_index);
            curr_bone_angular_velocities = db.bone_angular_velocities(frame_index);
            curr_bone_contacts = db.contact_states(frame_index);
        }
        
        // Update inertializer
        
        inertialize_pose_update(
            bone_positions,
            bone_velocities,
            bone_rotations,
            bone_angular_velocities,
            bone_offset_positions,
            bone_offset_velocities,
            bone_offset_rotations,
            bone_offset_angular_velocities,
            curr_bone_positions,
            curr_bone_velocities,
            curr_bone_rotations,
            curr_bone_angular_velocities,
            transition_src_position,
            transition_src_rotation,
            transition_dst_position,
            transition_dst_rotation,
            inertialize_blending_halflife,
            dt);
        
        // Update Simulation
        
        vec3 simulation_position_prev = simulation_position;
        vec3 animation_root_velocity_control = clamp_planar_speed(
            bone_velocities(0),
            db.root_speed_limit);
        vec3 simulation_desired_velocity = simulation_desired_velocity_update(
            desired_velocity,
            animation_root_velocity_control,
            simulation_follow_animation_speed,
            db.root_speed_limit);
        
        simulation_positions_update(
            simulation_position, 
            simulation_velocity, 
            simulation_acceleration,
            simulation_desired_velocity,
            simulation_velocity_halflife,
            dt,
            environment_boxes);
            
        simulation_rotations_update(
            simulation_rotation, 
            simulation_angular_velocity, 
            desired_rotation,
            simulation_rotation_halflife,
            dt);
        
        // Synchronization 
        
        if (synchronization_enabled)
        {
            vec3 synchronized_position = lerp(
                simulation_position, 
                bone_positions(0),
                synchronization_data_factor);
                
            quat synchronized_rotation = quat_nlerp_shortest(
                simulation_rotation,
                bone_rotations(0), 
                synchronization_data_factor);
          
            synchronized_position = simulation_collide_obstacles(
                simulation_position_prev,
                synchronized_position,
                environment_boxes);
            
            simulation_position = synchronized_position;
            simulation_rotation = synchronized_rotation;
            
            inertialize_root_adjust(
                bone_offset_positions(0),
                transition_src_position,
                transition_src_rotation,
                transition_dst_position,
                transition_dst_rotation,
                bone_positions(0),
                bone_rotations(0),
                synchronized_position,
                synchronized_rotation);
        }
        
        // Adjustment 
        
        if (!synchronization_enabled && adjustment_enabled)
        {   
            vec3 adjusted_position = bone_positions(0);
            quat adjusted_rotation = bone_rotations(0);
            
            if (adjustment_by_velocity_enabled)
            {
                adjusted_position = adjust_character_position_by_velocity(
                    bone_positions(0),
                    animation_root_velocity_control,
                    simulation_position,
                    adjustment_position_max_ratio,
                    adjustment_position_halflife,
                    dt);
                
                adjusted_rotation = adjust_character_rotation_by_velocity(
                    bone_rotations(0),
                    bone_angular_velocities(0),
                    simulation_rotation,
                    adjustment_rotation_max_ratio,
                    adjustment_rotation_halflife,
                    dt);
            }
            else
            {
                adjusted_position = adjust_character_position(
                    bone_positions(0),
                    simulation_position,
                    adjustment_position_halflife,
                    dt);
                
                adjusted_rotation = adjust_character_rotation(
                    bone_rotations(0),
                    simulation_rotation,
                    adjustment_rotation_halflife,
                    dt);
            }
      
            inertialize_root_adjust(
                bone_offset_positions(0),
                transition_src_position,
                transition_src_rotation,
                transition_dst_position,
                transition_dst_rotation,
                bone_positions(0),
                bone_rotations(0),
                adjusted_position,
                adjusted_rotation);
        }
        
        // Clamping
        
        if (!synchronization_enabled && clamping_enabled)
        {
            vec3 adjusted_position = bone_positions(0);
            quat adjusted_rotation = bone_rotations(0);
            
            adjusted_position = clamp_character_position(
                adjusted_position,
                simulation_position,
                clamping_max_distance);
            
            adjusted_rotation = clamp_character_rotation(
                adjusted_rotation,
                simulation_rotation,
                clamping_max_angle);
            
            inertialize_root_adjust(
                bone_offset_positions(0),
                transition_src_position,
                transition_src_rotation,
                transition_dst_position,
                transition_dst_rotation,
                bone_positions(0),
                bone_rotations(0),
                adjusted_position,
                adjusted_rotation);
        }

        bool startup_bootstrap_active =
            g_eval_config.enabled &&
            eval_frame_index < STARTUP_BOOTSTRAP_WINDOW_FRAMES;
        float startup_bootstrap_alpha =
            startup_bootstrap_active ?
            clampf((eval_frame_index + 1) / (float)STARTUP_BOOTSTRAP_WINDOW_FRAMES, 0.0f, 1.0f) :
            1.0f;

        if (startup_bootstrap_active)
        {
            float root_offset_y = bootstrap_support_root_offset(
                bone_positions,
                bone_rotations,
                curr_bone_contacts,
                contact_bones,
                db.bone_parents,
                environment_boxes,
                ik_foot_height);

            if (fabsf(root_offset_y) > 1e-4f)
            {
                vec3 bootstrap_position = bone_positions(0) + vec3(0.0f, root_offset_y, 0.0f);
                inertialize_root_adjust(
                    bone_offset_positions(0),
                    transition_src_position,
                    transition_src_rotation,
                    transition_dst_position,
                    transition_dst_rotation,
                    bone_positions(0),
                    bone_rotations(0),
                    bootstrap_position,
                    bone_rotations(0));
                simulation_position.y += root_offset_y;
            }
        }
        
        // Contact fixup with foot locking and IK

        adjusted_bone_positions = bone_positions;
        adjusted_bone_rotations = bone_rotations;

        if (ik_enabled)
        {
            for (int i = 0; i < contact_bones.size; i++)
            {
                // Find all the relevant bone indices
                int toe_bone = contact_bones(i);
                int heel_bone = db.bone_parents(toe_bone);
                int knee_bone = db.bone_parents(heel_bone);
                int hip_bone = db.bone_parents(knee_bone);
                int root_bone = db.bone_parents(hip_bone);
                
                // Compute the world space position for the toe
                global_bone_computed.zero();
                
                forward_kinematics_partial(
                    global_bone_positions,
                    global_bone_rotations,
                    global_bone_computed,
                    bone_positions,
                    bone_rotations,
                    db.bone_parents,
                    toe_bone);
                
                float toe_ground_height = environment_surface_height(
                    global_bone_positions(toe_bone).x,
                    global_bone_positions(toe_bone).z,
                    environment_boxes);
                vec3 toe_ground_normal = environment_surface_normal(
                    global_bone_positions(toe_bone).x,
                    global_bone_positions(toe_bone).z,
                    environment_boxes);

                if (startup_bootstrap_active)
                {
                    contact_bootstrap(
                        contact_states(i),
                        contact_locks(i),
                        contact_positions(i),
                        contact_velocities(i),
                        contact_points(i),
                        contact_targets(i),
                        contact_offset_positions(i),
                        contact_offset_velocities(i),
                        global_bone_positions(toe_bone),
                        curr_bone_contacts(i),
                        toe_ground_height,
                        ik_foot_height);
                }
                else
                {
                    contact_update(
                        contact_states(i),
                        contact_locks(i),
                        contact_positions(i),  
                        contact_velocities(i),
                        contact_points(i),
                        contact_targets(i),
                        contact_offset_positions(i),
                        contact_offset_velocities(i),
                        global_bone_positions(toe_bone),
                        curr_bone_contacts(i),
                        ik_unlock_radius,
                        toe_ground_height,
                        ik_foot_height,
                        ik_blending_halflife,
                        dt);
                }
                
                // Ensure contact position never goes through floor
                vec3 contact_position_clamp = contact_positions(i);
                contact_position_clamp.y = maxf(contact_position_clamp.y, toe_ground_height + ik_foot_height);
                if (startup_bootstrap_active)
                {
                    contact_position_clamp = lerp(
                        global_bone_positions(toe_bone),
                        contact_position_clamp,
                        startup_bootstrap_alpha);
                }
                
                // Re-compute toe, heel, knee, hip, and root bone positions
                for (int bone : {heel_bone, knee_bone, hip_bone, root_bone})
                {
                    forward_kinematics_partial(
                        global_bone_positions,
                        global_bone_rotations,
                        global_bone_computed,
                        bone_positions,
                        bone_rotations,
                        db.bone_parents,
                        bone);
                }
                
                // Perform simple two-joint IK to place heel
                ik_two_bone(
                    adjusted_bone_rotations(hip_bone),
                    adjusted_bone_rotations(knee_bone),
                    global_bone_positions(hip_bone),
                    global_bone_positions(knee_bone),
                    global_bone_positions(heel_bone),
                    contact_position_clamp + (global_bone_positions(heel_bone) - global_bone_positions(toe_bone)),
                    quat_mul_vec3(global_bone_rotations(knee_bone), vec3(0.0f, 1.0f, 0.0f)),
                    global_bone_rotations(hip_bone),
                    global_bone_rotations(knee_bone),
                    global_bone_rotations(root_bone),
                    ik_max_length_buffer);
                
                // Re-compute toe, heel, and knee positions 
                global_bone_computed.zero();
                
                for (int bone : {toe_bone, heel_bone, knee_bone})
                {
                    forward_kinematics_partial(
                        global_bone_positions,
                        global_bone_rotations,
                        global_bone_computed,
                        adjusted_bone_positions,
                        adjusted_bone_rotations,
                        db.bone_parents,
                        bone);
                }
                
                // Rotate heel so toe is facing toward contact point
                ik_look_at(
                    adjusted_bone_rotations(heel_bone),
                    global_bone_rotations(knee_bone),
                    global_bone_rotations(heel_bone),
                    global_bone_positions(heel_bone),
                    global_bone_positions(toe_bone),
                    contact_position_clamp);
                
                // Re-compute toe and heel positions
                global_bone_computed.zero();
                
                for (int bone : {toe_bone, heel_bone})
                {
                    forward_kinematics_partial(
                        global_bone_positions,
                        global_bone_rotations,
                        global_bone_computed,
                        adjusted_bone_positions,
                        adjusted_bone_rotations,
                        db.bone_parents,
                        bone);
                }
                
                // Rotate toe bone so that the end of the toe 
                // does not intersect with the ground
                vec3 toe_end_curr = quat_mul_vec3(
                    global_bone_rotations(toe_bone), vec3(ik_toe_length, 0.0f, 0.0f)) + 
                    global_bone_positions(toe_bone);
                vec3 toe_forward = quat_mul_vec3(global_bone_rotations(toe_bone), vec3(1.0f, 0.0f, 0.0f));
                vec3 toe_forward_tangent = toe_forward - toe_ground_normal * dot(toe_ground_normal, toe_forward);
                if (length(toe_forward_tangent) < 1e-4f)
                {
                    toe_forward_tangent = normalize(cross(vec3(0.0f, 1.0f, 0.0f), toe_ground_normal));
                }
                else
                {
                    toe_forward_tangent = normalize(toe_forward_tangent);
                }

                vec3 toe_end_targ = contact_position_clamp + toe_forward_tangent * ik_toe_length;
                float toe_end_ground_height = environment_surface_height(
                    toe_end_targ.x,
                    toe_end_targ.z,
                    environment_boxes);
                toe_end_targ.y = maxf(toe_end_targ.y, toe_end_ground_height + ik_foot_height);
                if (startup_bootstrap_active)
                {
                    toe_end_targ = lerp(toe_end_curr, toe_end_targ, startup_bootstrap_alpha);
                }
                
                ik_look_at(
                    adjusted_bone_rotations(toe_bone),
                    global_bone_rotations(heel_bone),
                    global_bone_rotations(toe_bone),
                    global_bone_positions(toe_bone),
                    toe_end_curr,
                    toe_end_targ);
            }
        }
        
        // Full pass of forward kinematics to compute 
        // all bone positions and rotations in the world
        // space ready for rendering
        
        forward_kinematics_full(
            global_bone_positions,
            global_bone_rotations,
            adjusted_bone_positions,
            adjusted_bone_rotations,
            db.bone_parents);

        float contact_height_error = 0.0f;
        float contact_slip = 0.0f;
        int terrain_penetrations = 0;
        int active_contacts = 0;

        for (int i = 0; i < contact_bones.size; i++)
        {
            int toe_bone = contact_bones(i);
            vec3 toe_pos = global_bone_positions(toe_bone);
            float toe_ground_height = environment_surface_height(
                toe_pos.x,
                toe_pos.z,
                environment_boxes);

            if (curr_bone_contacts(i))
            {
                active_contacts++;
                contact_height_error += fabsf(toe_pos.y - (toe_ground_height + ik_foot_height));
                if (toe_pos.y < toe_ground_height + ik_foot_height - 1e-3f)
                {
                    terrain_penetrations++;
                }
                if (eval_prev_contact_valid(i))
                {
                    vec3 delta = toe_pos - eval_prev_contact_positions(i);
                    contact_slip += sqrtf(squaref(delta.x) + squaref(delta.z)) / dt;
                }
                eval_prev_contact_valid(i) = true;
                eval_prev_contact_positions(i) = toe_pos;
            }
            else
            {
                eval_prev_contact_valid(i) = false;
            }
        }

        if (active_contacts > 0)
        {
            contact_height_error /= active_contacts;
            contact_slip /= active_contacts;
        }

        runtime_prev_contact_slip = contact_slip;
        runtime_prev_contact_height_error = contact_height_error;
        runtime_prev_terrain_penetrations = terrain_penetrations;
        bool severe_penetration =
            active_contacts > 0 &&
            (float)terrain_penetrations / (float)active_contacts > 0.5f &&
            contact_height_error > STARTUP_SEVERE_PENETRATION;
        runtime_emergency_reanchor =
            startup_bootstrap_active ?
            severe_penetration :
            (contact_slip > HYBRID_CONTACT_SLIP_HARD_RESET ||
             terrain_penetrations > 0 ||
             hybrid_residual_norm(residual_features, residual_latent) > HYBRID_RESIDUAL_NORM_MAX * 0.9f);

        if (g_eval_config.enabled && eval_trace != NULL)
        {

            runtime_debug_snapshot snapshot;
            snapshot.frame = eval_frame_index;
            snapshot.teacher_frame = teacher_frame_index;
            snapshot.selected_exemplar = exemplar_index;
            snapshot.exemplar_valid = exemplar_valid;
            snapshot.switch_flag = switch_flag;
            snapshot.residual_norm_before = residual_norm_before;
            snapshot.residual_norm_after = residual_norm_after;
            snapshot.stepper_input_norm = stepper_input_norm;
            snapshot.stepper_output_norm = stepper_output_norm;
            snapshot.decompressor_input_norm = decompressor_input_norm;
            snapshot.root_planar_speed = planar_speed(bone_velocities(0));
            snapshot.contact_slip = contact_slip;
            snapshot.correction_magnitude = projector_correction_mag;
            snapshot.switch_severity = switch_severity;
            snapshot.residual_carry_alpha = residual_carry_alpha;
            snapshot.selector_margin = selector_margin;
            snapshot.exemplar_dwell_frames = exemplar_dwell_frames;
            snapshot.emergency_reanchor = emergency_reanchor;
            snapshot.hard_reset = hard_reset;
            snapshot.finite_query = array_is_finite(query_normalized);
            snapshot.finite_features = array_is_finite(features_curr);
            snapshot.finite_latent = array_is_finite(latent_curr);
            snapshot.finite_residual = array_is_finite(residual_features) && array_is_finite(residual_latent);
            snapshot.finite_stepper = isfinite(stepper_input_norm) && isfinite(stepper_output_norm);
            snapshot.finite_decompressor = isfinite(decompressor_input_norm);
            snapshot.finite_pose = vec3_array_is_finite(bone_positions) && vec3_array_is_finite(bone_velocities);
            snapshot.topk_indices = copy_array_int(topk_indices);
            snapshot.topk_costs = copy_array(topk_costs);
            snapshot.selector_scores = copy_array(selector_scores);
            snapshot.query_normalized = copy_array(query_normalized);
            snapshot.features_curr = copy_array(features_curr);
            snapshot.latent_curr = copy_array(latent_curr);
            snapshot.residual_features = copy_array(residual_features);
            snapshot.residual_latent = copy_array(residual_latent);
            snapshot.stepper_control = copy_array(stepper_control);
            snapshot.divergence_flag =
                snapshot.root_planar_speed > db.root_speed_limit * 5.0f ||
                snapshot.contact_slip > 50.0f ||
                snapshot.correction_magnitude > 50.0f ||
                snapshot.residual_norm_after > HYBRID_RESIDUAL_NORM_MAX ||
                snapshot.stepper_input_norm > HYBRID_STEPPER_INPUT_NORM_MAX ||
                snapshot.decompressor_input_norm > 200.0f;

            failure_history.push_back(snapshot);
            if ((int)failure_history.size() > 30)
            {
                failure_history.erase(failure_history.begin());
            }

            bool frame_nonfinite =
                !snapshot.finite_query ||
                !snapshot.finite_features ||
                !snapshot.finite_latent ||
                !snapshot.finite_residual ||
                !snapshot.finite_stepper ||
                !snapshot.finite_decompressor ||
                !snapshot.finite_pose ||
                !isfinite(snapshot.root_planar_speed) ||
                !isfinite(snapshot.correction_magnitude) ||
                !isfinite(stepper_drift_mag) ||
                !isfinite(contact_height_error) ||
                !isfinite(contact_slip);

            if (!failure_triggered && (frame_nonfinite || snapshot.divergence_flag))
            {
                failure_triggered = true;
                failure_future_remaining = 5;
            }
            else if (failure_triggered && !failure_dump_written && failure_future_remaining > 0)
            {
                failure_future.push_back(snapshot);
                failure_future_remaining--;
            }

            if (failure_triggered && !failure_dump_written && failure_future_remaining == 0)
            {
                std::string dump_dir = g_eval_config.failure_dump_dir.empty() ? "." : g_eval_config.failure_dump_dir;
                std::string dump_path = dump_dir + "/" + g_eval_config.tag + "_failure_dump.txt";
                FILE* dump = fopen(dump_path.c_str(), "w");
                if (dump != NULL)
                {
                    fprintf(dump, "tag=%s selector_mode=%s residual_mode=%s\n", g_eval_config.tag.c_str(), g_eval_config.selector_mode.c_str(), g_eval_config.residual_mode.c_str());
                    fprintf(dump, "history\n");
                    for (const runtime_debug_snapshot& item : failure_history)
                    {
                        write_runtime_snapshot(dump, item);
                    }
                    fprintf(dump, "future\n");
                    for (const runtime_debug_snapshot& item : failure_future)
                    {
                        write_runtime_snapshot(dump, item);
                    }
                    fclose(dump);
                }
                failure_dump_written = true;
                eval_complete = true;
            }

            fprintf(
                eval_trace,
                "%d,%.6f,%.6f,%.6f,%.6f,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d,%.6f,%.6f,%.6f,%.6f,%.6f,",
                eval_frame_index,
                eval_frame_index * dt,
                bone_positions(0).x,
                bone_positions(0).y,
                bone_positions(0).z,
                teacher_frame_index,
                exemplar_index,
                exemplar_valid ? 1 : 0,
                switch_flag ? 1 : 0,
                snapshot.divergence_flag ? 1 : 0,
                residual_norm_before,
                residual_norm_after,
                stepper_input_norm,
                stepper_output_norm,
                decompressor_input_norm,
                snapshot.root_planar_speed,
                contact_height_error,
                contact_slip,
                terrain_penetrations,
                projector_correction_mag,
                stepper_drift_mag,
                switch_severity,
                residual_carry_alpha,
                selector_margin);
            fprintf(
                eval_trace,
                "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%s,%s,%s\n",
                exemplar_dwell_frames,
                emergency_reanchor ? 1 : 0,
                hard_reset ? 1 : 0,
                snapshot.finite_query ? 1 : 0,
                snapshot.finite_features ? 1 : 0,
                snapshot.finite_latent ? 1 : 0,
                snapshot.finite_residual ? 1 : 0,
                snapshot.finite_stepper ? 1 : 0,
                snapshot.finite_decompressor ? 1 : 0,
                snapshot.finite_pose ? 1 : 0,
                lmm_enabled ? "learned" : "ordinary",
                g_eval_config.selector_mode.c_str(),
                g_eval_config.residual_mode.c_str());

            eval_frame_index++;
            if (eval_frame_index >= g_eval_config.frames)
            {
                fclose(eval_trace);
                eval_trace = NULL;
                eval_complete = true;
                return;
            }
        }
        
        // Update camera
        
        if (terrain_preview_enabled)
        {
            vec3 preview_target = simulation_position;
            preview_target.y = environment_base_height(preview_target.x, preview_target.z);

            camera.position = (Vector3){ preview_target.x, preview_target.y + 18.0f, preview_target.z + 0.001f };
            camera.target = (Vector3){ preview_target.x, preview_target.y, preview_target.z };
            camera.up = (Vector3){ 0.0f, 0.0f, -1.0f };
            camera.fovy = 35.0f;
        }
        else
        {
            camera.up = (Vector3){ 0.0f, 1.0f, 0.0f };
            camera.fovy = 45.0f;

            orbit_camera_update(
                camera, 
                camera_azimuth,
                camera_altitude,
                camera_distance,
                bone_positions(0) + vec3(0, 1, 0),
                gamepadstick_right,
                desired_strafe,
                dt);
        }

        if (g_eval_config.enabled)
        {
            return;
        }

        // Render
        
        BeginDrawing();
        ClearBackground(RAYWHITE);
        
        BeginMode3D(camera);
        
        // Draw Simulation Object
        
        DrawCylinderWires(to_Vector3(simulation_position), 0.6f, 0.6f, 0.001f, 17, ORANGE);
        DrawSphereWires(to_Vector3(simulation_position), 0.05f, 4, 10, ORANGE);
        DrawLine3D(to_Vector3(simulation_position), to_Vector3(
            simulation_position + 0.6f * quat_mul_vec3(simulation_rotation, vec3(0.0f, 0.0f, 1.0f))), ORANGE);
        
        // Draw Clamping Radius/Angles
        
        if (clamping_enabled)
        {
            DrawCylinderWires(
                to_Vector3(simulation_position), 
                clamping_max_distance, 
                clamping_max_distance, 
                0.001f, 17, SKYBLUE);
            
            quat rotation_clamp_0 = quat_mul(quat_from_angle_axis(+clamping_max_angle, vec3(0.0f, 1.0f, 0.0f)), simulation_rotation);
            quat rotation_clamp_1 = quat_mul(quat_from_angle_axis(-clamping_max_angle, vec3(0.0f, 1.0f, 0.0f)), simulation_rotation);
            
            vec3 rotation_clamp_0_dir = simulation_position + 0.6f * quat_mul_vec3(rotation_clamp_0, vec3(0.0f, 0.0f, 1.0f));
            vec3 rotation_clamp_1_dir = simulation_position + 0.6f * quat_mul_vec3(rotation_clamp_1, vec3(0.0f, 0.0f, 1.0f));

            DrawLine3D(to_Vector3(simulation_position), to_Vector3(rotation_clamp_0_dir), SKYBLUE);
            DrawLine3D(to_Vector3(simulation_position), to_Vector3(rotation_clamp_1_dir), SKYBLUE);
        }
        
        // Draw IK foot lock positions
        
        if (ik_enabled)
        {
            for (int i = 0; i <  contact_positions.size; i++)
            {
                if (contact_locks(i))
                {
                    DrawSphereWires(to_Vector3(contact_positions(i)), 0.05f, 4, 10, PINK);
                }
            }
        }
        
        draw_trajectory(
            trajectory_positions,
            trajectory_rotations,
            ORANGE);
        
        draw_environment_boxes(environment_boxes);
        
        if (debug_draw_mesh)
        {
            deform_character_mesh(
                character_mesh, 
                character_data, 
                debug_draw_bind_mesh ? slice1d<vec3>(character_data.bone_rest_positions) : slice1d<vec3>(global_bone_positions), 
                debug_draw_bind_mesh ? slice1d<quat>(character_data.bone_rest_rotations) : slice1d<quat>(global_bone_rotations),
                db.bone_parents);
            
            DrawModel(character_model, (Vector3){0.0f, 0.0f, 0.0f}, 1.0f, RAYWHITE);
        }

        if (debug_draw_skeleton)
        {
            draw_skeleton(global_bone_positions, db.bone_parents, DARKGRAY, BLACK);
        }

        if (debug_draw_bind_skeleton)
        {
            draw_skeleton(
                character_data.bone_rest_positions,
                db.bone_parents,
                BLUE,
                ORANGE,
                vec3(0.35f, 0.0f, 0.0f));
        }
        
        // Draw current query features so the trajectory/environment preview
        // stays continuous even when the matched frame changes discretely.
        draw_features(debug_query_features, bone_positions(0), bone_rotations(0), MAROON);
        
        // Draw Simuation Bone
        
        DrawSphereWires(to_Vector3(bone_positions(0)), 0.05f, 4, 10, MAROON);
        DrawLine3D(to_Vector3(bone_positions(0)), to_Vector3(
            bone_positions(0) + 0.6f * quat_mul_vec3(bone_rotations(0), vec3(0.0f, 0.0f, 1.0f))), MAROON);
        
        // Draw Ground Plane
        
        DrawModel(ground_plane_model, (Vector3){0.0f, -0.01f, 0.0f}, 1.0f, WHITE);
        DrawGrid(20, 1.0f);
        draw_axis(vec3(), quat());
        
        EndMode3D();

        // UI
        
        //---------
        
        float ui_sim_hei = 20;
        
        GuiGroupBox((Rectangle){ 970, ui_sim_hei, 290, 250 }, "simulation object");
        GuiCheckBox(
            (Rectangle){ 990, ui_sim_hei + 10, 20, 20 },
            "follow anim speed",
            &simulation_follow_animation_speed);

        GuiSliderBar(
            (Rectangle){ 1100, ui_sim_hei + 10, 120, 20 }, 
            "velocity halflife", 
            TextFormat("%5.3f", simulation_velocity_halflife), 
            &simulation_velocity_halflife, 0.0f, 0.5f);
            
        GuiSliderBar(
            (Rectangle){ 1100, ui_sim_hei + 40, 120, 20 }, 
            "rotation halflife", 
            TextFormat("%5.3f", simulation_rotation_halflife), 
            &simulation_rotation_halflife, 0.0f, 0.5f);
            
        GuiSliderBar(
            (Rectangle){ 1100, ui_sim_hei + 70, 120, 20 }, 
            "run forward request", 
            TextFormat("%5.3f", simulation_run_fwrd_speed), 
            &simulation_run_fwrd_speed, 0.0f, 10.0f);
        
        GuiSliderBar(
            (Rectangle){ 1100, ui_sim_hei + 100, 120, 20 }, 
            "run sideways request", 
            TextFormat("%5.3f", simulation_run_side_speed), 
            &simulation_run_side_speed, 0.0f, 10.0f);
        
        GuiSliderBar(
            (Rectangle){ 1100, ui_sim_hei + 130, 120, 20 }, 
            "run backwards request", 
            TextFormat("%5.3f", simulation_run_back_speed), 
            &simulation_run_back_speed, 0.0f, 10.0f);
        
        GuiSliderBar(
            (Rectangle){ 1100, ui_sim_hei + 160, 120, 20 }, 
            "walk forward request", 
            TextFormat("%5.3f", simulation_walk_fwrd_speed), 
            &simulation_walk_fwrd_speed, 0.0f, 5.0f);
        
        GuiSliderBar(
            (Rectangle){ 1100, ui_sim_hei + 190, 120, 20 }, 
            "walk sideways request", 
            TextFormat("%5.3f", simulation_walk_side_speed), 
            &simulation_walk_side_speed, 0.0f, 5.0f);
        
        GuiSliderBar(
            (Rectangle){ 1100, ui_sim_hei + 220, 120, 20 }, 
            "walk backwards request", 
            TextFormat("%5.3f", simulation_walk_back_speed), 
            &simulation_walk_back_speed, 0.0f, 5.0f);
        
        //---------
        
        float ui_inert_hei = 280;
        
        GuiGroupBox((Rectangle){ 970, ui_inert_hei, 290, 40 }, "inertiaization blending");
        
        GuiSliderBar(
            (Rectangle){ 1100, ui_inert_hei + 10, 120, 20 }, 
            "halflife", 
            TextFormat("%5.3f", inertialize_blending_halflife), 
            &inertialize_blending_halflife, 0.0f, 0.3f);
        
        //---------
        
        float ui_lmm_hei = 330;
        
        GuiGroupBox((Rectangle){ 970, ui_lmm_hei, 290, 65 }, "learned motion matching");
        
        GuiCheckBox(
            (Rectangle){ 1000, ui_lmm_hei + 10, 20, 20 }, 
            "enabled",
            &lmm_enabled);
        GuiSliderBar(
            (Rectangle){ 1110, ui_lmm_hei + 10, 110, 20 },
            "threshold",
            TextFormat("%5.3f", lmm_projector_transition_threshold),
            &lmm_projector_transition_threshold, 0.0f, 0.5f);
        GuiSliderBar(
            (Rectangle){ 1110, ui_lmm_hei + 35, 110, 20 },
            "blend halflife",
            TextFormat("%5.3f", lmm_projector_blend_halflife),
            &lmm_projector_blend_halflife, 0.01f, 0.3f);
        if (!lmm_networks_compatible)
        {
            GuiLabel((Rectangle){ 1065, ui_lmm_hei + 10, 185, 20 }, "retrain nets for env sdf");
            lmm_enabled = false;
        }
        
        //---------
        
        float ui_ctrl_hei = 380;
        
        GuiGroupBox((Rectangle){ 1010, ui_ctrl_hei, 250, 260 }, "controls");
        GuiLabel((Rectangle){ 1030, ui_ctrl_hei - 20, 210, 20 }, PFNN_HEIGHTMAP_LABEL);
        GuiLabel((Rectangle){ 1030, ui_ctrl_hei +  20, 200, 20 }, terrain_preview_enabled ? "T Topdown Preview: ON" : "T Topdown Preview: OFF");
        
        GuiLabel((Rectangle){ 1030, ui_ctrl_hei +  40, 200, 20 }, "Left Trigger - Strafe");
        GuiLabel((Rectangle){ 1030, ui_ctrl_hei +  60, 200, 20 }, "WASD / Left Stick - Move");
        GuiLabel((Rectangle){ 1030, ui_ctrl_hei +  80, 200, 20 }, "RMB / Arrows / Right Stick - Camera");
        GuiLabel((Rectangle){ 1030, ui_ctrl_hei + 100, 200, 20 }, "Left Shoulder - Zoom In");
        GuiLabel((Rectangle){ 1030, ui_ctrl_hei + 120, 200, 20 }, "Right Shoulder - Zoom Out");
        GuiLabel((Rectangle){ 1030, ui_ctrl_hei + 140, 200, 20 }, "A Button - Walk");
        GuiLabel((Rectangle){ 1030, ui_ctrl_hei + 160, 200, 20 }, debug_draw_mesh ? "M Mesh: ON" : "M Mesh: OFF");
        GuiLabel((Rectangle){ 1030, ui_ctrl_hei + 180, 200, 20 }, debug_draw_skeleton ? "B Skeleton: ON" : "B Skeleton: OFF");
        GuiLabel((Rectangle){ 1030, ui_ctrl_hei + 200, 200, 20 }, ik_enabled ? "I IK: ON" : "I IK: OFF");
        GuiLabel((Rectangle){ 1030, ui_ctrl_hei + 220, 200, 20 }, debug_draw_bind_skeleton ? "N Bind Skeleton: ON" : "N Bind Skeleton: OFF");
        GuiLabel((Rectangle){ 1030, ui_ctrl_hei + 240, 200, 20 }, debug_draw_bind_mesh ? "R Bind Mesh: ON" : "R Bind Mesh: OFF");
        

        
        //---------
        
        GuiGroupBox((Rectangle){ 20, 20, 290, 220 }, "feature weights");
        
        GuiSliderBar(
            (Rectangle){ 150, 30, 120, 20 }, 
            "foot position", 
            TextFormat("%5.3f", feature_weight_foot_position), 
            &feature_weight_foot_position, 0.001f, 3.0f);
            
        GuiSliderBar(
            (Rectangle){ 150, 60, 120, 20 }, 
            "foot velocity", 
            TextFormat("%5.3f", feature_weight_foot_velocity), 
            &feature_weight_foot_velocity, 0.001f, 3.0f);
        
        GuiSliderBar(
            (Rectangle){ 150, 90, 120, 20 }, 
            "hip velocity", 
            TextFormat("%5.3f", feature_weight_hip_velocity), 
            &feature_weight_hip_velocity, 0.001f, 3.0f);
        
        GuiSliderBar(
            (Rectangle){ 150, 120, 120, 20 }, 
            "trajectory positions", 
            TextFormat("%5.3f", feature_weight_trajectory_positions), 
            &feature_weight_trajectory_positions, 0.001f, 3.0f);
        
        GuiSliderBar(
            (Rectangle){ 150, 150, 120, 20 }, 
            "trajectory directions", 
            TextFormat("%5.3f", feature_weight_trajectory_directions), 
            &feature_weight_trajectory_directions, 0.001f, 3.0f);
        GuiSliderBar(
            (Rectangle){ 150, 180, 120, 20 },
            "environment",
            TextFormat("%5.3f", feature_weight_environment),
            &feature_weight_environment, 0.001f, 3.0f);
            
        if (GuiButton((Rectangle){ 150, 210, 120, 20 }, "rebuild database"))
        {
            rebuild_matching_database();
        }
        
        //---------
        
        float ui_sync_hei = 220;
        
        GuiGroupBox((Rectangle){ 20, ui_sync_hei, 290, 70 }, "synchronization");

        GuiCheckBox(
            (Rectangle){ 50, ui_sync_hei + 10, 20, 20 }, 
            "enabled",
            &synchronization_enabled);

        GuiSliderBar(
            (Rectangle){ 150, ui_sync_hei + 40, 120, 20 }, 
            "data-driven amount", 
            TextFormat("%5.3f", synchronization_data_factor), 
            &synchronization_data_factor, 0.0f, 1.0f);

        //---------
        
        float ui_adj_hei = 300;
        
        GuiGroupBox((Rectangle){ 20, ui_adj_hei, 290, 130 }, "adjustment");
        
        GuiCheckBox(
            (Rectangle){ 50, ui_adj_hei + 10, 20, 20 }, 
            "enabled",
            &adjustment_enabled);    
        
        GuiCheckBox(
            (Rectangle){ 50, ui_adj_hei + 40, 20, 20 }, 
            "clamp to max velocity",
            &adjustment_by_velocity_enabled);    
        
        GuiSliderBar(
            (Rectangle){ 150, ui_adj_hei + 70, 120, 20 }, 
            "position halflife", 
            TextFormat("%5.3f", adjustment_position_halflife), 
            &adjustment_position_halflife, 0.0f, 0.5f);
        
        GuiSliderBar(
            (Rectangle){ 150, ui_adj_hei + 100, 120, 20 }, 
            "rotation halflife", 
            TextFormat("%5.3f", adjustment_rotation_halflife), 
            &adjustment_rotation_halflife, 0.0f, 0.5f);
        
        //---------
        
        float ui_clamp_hei = 440;
        
        GuiGroupBox((Rectangle){ 20, ui_clamp_hei, 290, 100 }, "clamping");
        
        GuiCheckBox(
            (Rectangle){ 50, ui_clamp_hei + 10, 20, 20 }, 
            "enabled",
            &clamping_enabled);      
        
        GuiSliderBar(
            (Rectangle){ 150, ui_clamp_hei + 40, 120, 20 }, 
            "distance", 
            TextFormat("%5.3f", clamping_max_distance), 
            &clamping_max_distance, 0.0f, 0.5f);
        
        GuiSliderBar(
            (Rectangle){ 150, ui_clamp_hei + 70, 120, 20 }, 
            "angle", 
            TextFormat("%5.3f", clamping_max_angle), 
            &clamping_max_angle, 0.0f, PIf);
        
        //---------
        
        float ui_ik_hei = 550;
        
        GuiGroupBox((Rectangle){ 20, ui_ik_hei, 290, 100 }, "inverse kinematics");
        
        bool ik_enabled_prev = ik_enabled;
        
        GuiCheckBox(
            (Rectangle){ 50, ui_ik_hei + 10, 20, 20 }, 
            "enabled",
            &ik_enabled);      
        
        // Foot locking needs resetting when IK is toggled
        if (ik_enabled && !ik_enabled_prev)
        {
            for (int i = 0; i < contact_bones.size; i++)
            {
                vec3 bone_position;
                vec3 bone_velocity;
                quat bone_rotation;
                vec3 bone_angular_velocity;
                
                forward_kinematics_velocity(
                    bone_position,
                    bone_velocity,
                    bone_rotation,
                    bone_angular_velocity,
                    bone_positions,
                    bone_velocities,
                    bone_rotations,
                    bone_angular_velocities,
                    db.bone_parents,
                    contact_bones(i));
                
                contact_reset(
                    contact_states(i),
                    contact_locks(i),
                    contact_positions(i),  
                    contact_velocities(i),
                    contact_points(i),
                    contact_targets(i),
                    contact_offset_positions(i),
                    contact_offset_velocities(i),
                    bone_position,
                    bone_velocity,
                    false);
            }
        }
        
        GuiSliderBar(
            (Rectangle){ 150, ui_ik_hei + 40, 120, 20 }, 
            "blending halflife", 
            TextFormat("%5.3f", ik_blending_halflife), 
            &ik_blending_halflife, 0.0f, 1.0f);
        
        GuiSliderBar(
            (Rectangle){ 150, ui_ik_hei + 70, 120, 20 }, 
            "unlock radius", 
            TextFormat("%5.3f", ik_unlock_radius), 
            &ik_unlock_radius, 0.0f, 0.5f);
        
        //---------

        EndDrawing();

    };

#if defined(PLATFORM_WEB)
    std::function<void()> u{update_func};
    emscripten_set_main_loop_arg(update_callback, &u, 0, 1);
#else
    while (!WindowShouldClose() && !eval_complete)
    {
        update_func();
    }
#endif

    // Unload stuff and finish
    if (!g_eval_config.enabled)
    {
        UnloadModel(character_model);
        UnloadModel(ground_plane_model);
        UnloadShader(character_shader);
        UnloadShader(ground_plane_shader);
    }
    if (eval_trace != NULL)
    {
        fclose(eval_trace);
        eval_trace = NULL;
    }

    CloseWindow();

    return 0;
}
