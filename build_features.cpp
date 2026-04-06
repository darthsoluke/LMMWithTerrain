#include "character.h"
#include "database.h"
#include <stdio.h>

int main(int argc, char** argv)
{
    const char* database_path = argc > 1 ? argv[1] : "resources/database.bin";
    const char* environment_path = argc > 2 ? argv[2] : "resources/terrain_features.bin";
    const char* output_path = argc > 3 ? argv[3] : "resources/features.bin";

    database db;
    printf("Loading %s...\n", database_path);
    database_load(db, database_path);

    float feature_weight_foot_position = 0.75f;
    float feature_weight_foot_velocity = 1.0f;
    float feature_weight_hip_velocity = 1.0f;
    float feature_weight_trajectory_positions = 1.0f;
    float feature_weight_trajectory_directions = 1.5f;
    float feature_weight_environment = 1.0f;

    printf("Building matching features using %s...\n", environment_path);
    database_build_matching_features(
        db,
        feature_weight_foot_position,
        feature_weight_foot_velocity,
        feature_weight_hip_velocity,
        feature_weight_trajectory_positions,
        feature_weight_trajectory_directions,
        feature_weight_environment,
        environment_path);

    printf("Saving %s...\n", output_path);
    database_save_matching_features(db, output_path);
    printf("Done. nframes=%d nfeatures=%d\n", db.nframes(), db.nfeatures());
    return 0;
}
