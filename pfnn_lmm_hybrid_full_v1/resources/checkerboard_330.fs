#version 330

in vec3 fragPosition;
in vec2 fragTexCoord;
in vec4 fragColor;
in vec3 fragNormal;

uniform vec4 colDiffuse;

out vec4 finalColor;

void main()
{
    float total = floor(fragPosition.x * 2.0) +
                  floor(fragPosition.z * 2.0);

    vec3 checker = mod(total, 2.0) == 0.0 ?
        vec3(0.70, 0.72, 0.70) :
        vec3(0.82, 0.84, 0.82);

    float height_t = clamp((fragPosition.y + 8.0) / 20.0, 0.0, 1.0);
    vec3 low_color = vec3(0.18, 0.42, 0.28);
    vec3 high_color = vec3(0.82, 0.78, 0.62);
    vec3 height_color = mix(low_color, high_color, height_t);

    vec3 light_dir = normalize(vec3(-0.45, 1.0, -0.30));
    float diffuse = clamp(dot(normalize(fragNormal), light_dir), 0.0, 1.0);
    float lighting = 0.35 + 0.65 * diffuse;

    vec3 color = mix(checker, height_color, 0.55) * lighting;
    finalColor = vec4(color, 1.0);
}
