#version 300 es
///fShader///
/*
* v1.2
* last change in Haunting The Hauntessa
*/

#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
precision highp sampler3D;
#else
precision mediump float;
precision mediump sampler3D;
#endif

struct Material {
    vec3 ambientColor;
    vec3 diffuseColor;
    vec3 specularColor;
    float shininess;
};

const int N_LIGHTS = 1; // replaced before compiling
float illumination;
uniform vec3 uPointLights[N_LIGHTS];
uniform vec3 uLightColors[N_LIGHTS];
uniform vec3 uLightDirections[N_LIGHTS];
uniform sampler2D uSampler;
uniform vec3 uCameraPos;
uniform Material uMaterial;
uniform sampler3D uOcclusionMap;   // Occlusion map
uniform vec3 uGridSize;            // Dimensions of the 3D occlusion map in world space

in vec3 FragPos;
in vec3 v_normal;
in vec2 vTextureCoord;

// --- TUNABLE CONSTANTS ---
const vec3 innerLightColor = vec3(1.0f, 1.0f, 1.0f);
const float innerAmbientStrength = 0.30f; 
const float innerDiffuseStrength = 15.0f;
const float innerSpecularStrength = 5.0f;

const float PL_AmbientStrength = 5.0f;
const float PL_DiffuseStrength = 10.0f;
const float PL_SpecularStrength = 2.5f;

const float IGNORE_ALPHA = 0.2f;
const int   MAX_STEPS = 99;
const float EPSILON = 0.005f;
const float PL_AMBIENT_OCCLUSION = 0.225f;
const float PL_DIFFUSE_OCCLUSION = 0.30f;
const float PL_AMBIENT_ILLUMINATION_REDUCTION = 0.02f;
const float PL_DIFUSSE_ILLUMINATION_REDUCTION = 0.20f;
const float PL_DIFUSSE_LIGHT_HALO_REDUCTION = 0.40f;
const float ATTNF = 0.3f;
const float ATTNF2 = 0.75f;
const float HATTNF = 1.5f;
const float HATTNF2 = 5.0f;
const float MAXLIGHT = 0.90f;
const float IGNORED_ATTN_DISTANCE = 0.012f;
const float ILLUMINATION_CUTOFF = 0.11f;
const float DISTANCE_LIGHT = 0.475f;
const float LIGHT_POS_Y_OFFSET = 0.35f;
const float INTO_WALL = 0.01f;

out vec4 fragColor;

// ----------------------------------------------------------------------------
// Function prototypes
// ----------------------------------------------------------------------------
vec3 CalcLight(
    vec3 lightPosition, vec3 FragPos, vec3 viewDir, vec3 normal,
    vec3 pointLightColor, float shininess, vec3 ambientColor,
    vec3 diffuseColor, vec3 specularColor, float ambientStrength,
    float diffuseStrength, float specularStrength, int inner,
    vec3 lightDirection
);

bool Raycast3D(vec3 rayOrigin3D, vec3 rayTarget3D, float illumination);
bool isOccluded(vec3 position3D);
vec3 worldToNormalizedTexCoord3D(vec3 position3D);

// ----------------------------------------------------------------------------

void main(void) {
    // Basic material properties
    vec3 ambientColor = uMaterial.ambientColor;
    vec3 diffuseColor = uMaterial.diffuseColor;
    vec3 specularColor = uMaterial.specularColor;
    float shininess = uMaterial.shininess;

    // Prepare normal & view direction
    vec3 norm = normalize(v_normal);
    vec3 viewDir = normalize(uCameraPos - FragPos);

    // "Inner" light from the camera position
    vec3 innerLight = CalcLight(
        uCameraPos, FragPos, viewDir, norm,
        innerLightColor, shininess,
        ambientColor, diffuseColor, specularColor,
        innerAmbientStrength, innerDiffuseStrength, innerSpecularStrength,
        1, viewDir
    );

    // Sum point lights
    vec3 PL_output = vec3(0.0f);
    for (int i = 0; i < N_LIGHTS; i++) {
        if (uPointLights[i].x < 0.0f) {
            continue;
        }
        PL_output += CalcLight(
            uPointLights[i], FragPos, viewDir, norm,
            uLightColors[i], shininess,
            ambientColor, diffuseColor, specularColor,
            PL_AmbientStrength, PL_DiffuseStrength, PL_SpecularStrength,
            0, uLightDirections[i]
        );
    }

    vec3 light = innerLight + PL_output;

    vec4 texelColor = texture(uSampler, vTextureCoord);
    if (texelColor.a < IGNORE_ALPHA) {
        discard;
    }

    fragColor = vec4(texelColor.rgb * light, texelColor.a);
}

// ----------------------------------------------------------------------------
// Lighting Calculation
// ----------------------------------------------------------------------------
vec3 CalcLight(
    vec3 lightPosition, vec3 FragPos, vec3 viewDir, vec3 normal,
    vec3 pointLightColor, float shininess,
    vec3 ambientColor, vec3 diffuseColor, vec3 specularColor,
    float ambientStrength, float diffuseStrength, float specularStrength,
    int inner, vec3 lightDirection
) {
    float lightPosDistance = distance(lightPosition, FragPos);
    float lightDistance = distance(
        vec3(lightPosition.x, lightPosition.y - LIGHT_POS_Y_OFFSET, lightPosition.z),
        FragPos
    );
    vec3 lightDir = normalize(FragPos - lightPosition);
    vec3 directionOfOrthoLight = lightDirection; // assumed normalized

    float invDistance = 1.0f / lightPosDistance;
    float attenuation = invDistance / (ATTNF + ATTNF2 * lightPosDistance);

    illumination = 1.0f;
    // For “directional” lights, your code checks if x < 128.0 => directional
    if (inner == 0 && lightDirection.x < 128.0f) {
        illumination = dot(lightDir, directionOfOrthoLight);
    }

    bool occluded = Raycast3D(lightPosition, FragPos, illumination);

    bool isLight = false;
    if (lightDistance < DISTANCE_LIGHT) {
        isLight = true;
    }

    // Ambient
    vec3 ambientLight = (inner == 1)
        ? (pointLightColor * ambientStrength * ambientColor)
        : (pointLightColor * ambientStrength * attenuation * ambientColor);

    // Diffuse 
    float diffLight = max(dot(normal, lightDir), 0.0f);
    float diffView  = max(dot(normal, viewDir), 0.0f);
    float diff      = 0.95f * diffLight + 0.05f * diffView;
    vec3 diffuselight = pointLightColor * diff * diffuseStrength * attenuation * diffuseColor;

    // Specular
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec      = pow(max(dot(viewDir, reflectDir), 0.0f), shininess);
    vec3 specularLight = pointLightColor * spec * specularStrength * attenuation * specularColor;

    // Clamp
    ambientLight  = clamp(ambientLight,  0.0f, MAXLIGHT);
    diffuselight  = clamp(diffuselight,  0.0f, MAXLIGHT);
    specularLight = clamp(specularLight, 0.0f, MAXLIGHT);

    // Light cutoff / halo logic
    if (illumination < ILLUMINATION_CUTOFF) {
        if (isLight) {
            float invlightDistance = 1.0f / lightDistance;
            float attenuationHalo  = invlightDistance / (HATTNF + HATTNF2 * lightDistance);
            diffuselight *= PL_DIFUSSE_LIGHT_HALO_REDUCTION * attenuationHalo;
        } else {
            diffuselight *= PL_DIFUSSE_ILLUMINATION_REDUCTION;
        }
        if (lightPosDistance > IGNORED_ATTN_DISTANCE) {
            ambientLight *= PL_AMBIENT_ILLUMINATION_REDUCTION;
        }
    }
    // Occlusion
    else if (occluded && inner == 0) {
        return PL_AMBIENT_OCCLUSION * ambientLight + PL_DIFFUSE_OCCLUSION * diffuselight;
    }

    return ambientLight + diffuselight + specularLight;
}

// ----------------------------------------------------------------------------
// 3D Raycasting (Same-cell approach)
// ----------------------------------------------------------------------------
bool Raycast3D(vec3 rayOrigin3D, vec3 rayTarget3D, float illumination) {
    // We still set up direction and the adjusted target, in case we want
    // to nudge the endpoint into a wall by a small offset:
    vec3 direction   = rayTarget3D - rayOrigin3D;   
    float illumination2 = (illumination + EPSILON) * (illumination + EPSILON);
    vec3 adjustedTarget = rayTarget3D - normalize(direction) * INTO_WALL * illumination2;
    direction           = adjustedTarget - rayOrigin3D;

    // *** Simplification: We'll floor() the target and see if we reach that cell. ***
    vec3 cellTarget = floor(adjustedTarget);

    // Step direction in each axis: +1 or -1
    vec3 step = sign(direction);

    // Avoid dividing by zero
    vec3 invDirection = 1.0f / max(abs(direction), vec3(EPSILON));

    // tDelta is how far in param space we go to cross each boundary in x, y, or z.
    vec3 tDelta = invDirection;

    // Identify our start cell (integer coordinates).
    vec3 startCell = floor(rayOrigin3D);
    vec3 cellOffset = rayOrigin3D - startCell;

    // Compute tMax, how far (in param t) until we hit the first boundary along each axis.
    vec3 tMax = vec3(0.0);
    for (int i = 0; i < 3; i++) {
        float distToBoundary;
        if (step[i] > 0.0) {
            distToBoundary = 1.0 - fract(cellOffset[i]);
        } else {
            distToBoundary = fract(cellOffset[i]);
        }
        // If offset is effectively zero, we might be on the boundary => ensure non-zero
        if (distToBoundary == 0.0) {
            distToBoundary = 1.0;
        }
        tMax[i] = distToBoundary * tDelta[i];
    }

    // Current cell position in integer coordinates
    vec3 currentCell = startCell;

    // *** Walk along the ray in 3D, up to MAX_STEPS ***
    for(int i = 0; i < MAX_STEPS; i++) {

        // 1) Check if we are in the same cell as the (floored) target
        if (currentCell == cellTarget) {
            // Means we have arrived at or passed the target cell
            return false;
        }

        // 2) Check if this cell is occluded
        vec3 cellCenter = currentCell + vec3(0.5);
        if (isOccluded(cellCenter)) {
            return true;
        }

        // 3) Step to the next boundary
        // We see which axis is smallest in tMax => that boundary is next
        if(tMax.x < tMax.y) {
            if(tMax.x < tMax.z) {
                // Cross in x
                tMax.x += tDelta.x;
                currentCell.x += step.x;
            }
            else {
                // Cross in z
                tMax.z += tDelta.z;
                currentCell.z += step.z;
            }
        }
        else {
            if(tMax.y < tMax.z) {
                // Cross in y
                tMax.y += tDelta.y;
                currentCell.y += step.y;
            }
            else {
                // Cross in z
                tMax.z += tDelta.z;
                currentCell.z += step.z;
            }
        }
    }

    // If we never found occlusion within MAX_STEPS, assume no occlusion
    return false;
}

// ----------------------------------------------------------------------------
// 3D Texture Helpers
// ----------------------------------------------------------------------------
bool isOccluded(vec3 position3D) {
    // Convert from world position to [0,1] texture coordinates
    vec3 texCoord = worldToNormalizedTexCoord3D(position3D);
    float occlusion = texture(uOcclusionMap, texCoord).r;
    return (occlusion > 0.5);
}

vec3 worldToNormalizedTexCoord3D(vec3 position3D) {
    // Normalize x,y,z by the 3D grid size
    return vec3(
        position3D.x / uGridSize.x,
        position3D.y / uGridSize.y,
        position3D.z / uGridSize.z
    );
}
