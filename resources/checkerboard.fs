#version 300 es
precision mediump float;

in vec3 fragPosition;
in vec2 fragTexCoord;
in vec4 fragColor;
in vec3 fragNormal;

uniform vec4 colDiffuse;

out vec4 finalColor;

void main()
{
    float total = floor(fragPosition.x * 2.0f) +
                  floor(fragPosition.z * 2.0f);

    vec3 checker = mod(total, 2.0f) == 0.0f ?
        vec3(0.70f, 0.72f, 0.70f) :
        vec3(0.82f, 0.84f, 0.82f);

    float height_t = clamp((fragPosition.y + 8.0f) / 20.0f, 0.0f, 1.0f);
    vec3 low_color = vec3(0.18f, 0.42f, 0.28f);
    vec3 high_color = vec3(0.82f, 0.78f, 0.62f);
    vec3 height_color = mix(low_color, high_color, height_t);

    vec3 light_dir = normalize(vec3(-0.45f, 1.0f, -0.30f));
    float diffuse = clamp(dot(normalize(fragNormal), light_dir), 0.0f, 1.0f);
    float lighting = 0.35f + 0.65f * diffuse;

    vec3 color = mix(checker, height_color, 0.55f) * lighting;
    finalColor = vec4(color, 1.0f);
}
