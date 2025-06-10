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

const int N_LIGHTS = 1;                                         //replaced before compiling
float illumination;
uniform vec3 uPointLights[N_LIGHTS];
uniform vec3 uLightColors[N_LIGHTS];
uniform vec3 uLightDirections[N_LIGHTS];
uniform sampler2D uSampler;
uniform vec3 uCameraPos;
uniform Material uMaterial;
uniform sampler3D uOcclusionMap;                                // Occlusion map
uniform vec3 uGridSize;                                         // Size of the grid in the occlusion map

in vec3 FragPos;
in vec3 v_normal;
in vec2 vTextureCoord;

//bloody hardcoded constants
const vec3 innerLightColor = vec3(1.0f, 1.0f, 1.0f);
const float innerAmbientStrength = 0.30f;                        //0.30
const float innerDiffuseStrength = 25.0f;                        //20.0
const float innerSpecularStrength = 5.0f;                        //5.0

const float PL_AmbientStrength = 9.99f;                           //9.99
const float PL_DiffuseStrength = 50.0f;                          //50.0
const float PL_SpecularStrength = 5.0f;                          //2.5

const float IGNORE_ALPHA = 0.1f;
const int MAX_STEPS = 999;                                       // Max steps for raycasting loop - 99
const float EPSILON = 0.005f;                                    // don't enter the wall, check for occlusion - 0.005
const float PL_AMBIENT_OCCLUSION = 0.10f;                        // how much of ambient light gets through occlusion - 0.225
const float PL_DIFFUSE_OCCLUSION = 0.10f;                        // how much of diffused light gets through occlusion - 0.30
const float PL_AMBIENT_ILLUMINATION_REDUCTION = 0.02f;           // how much of ambient light gets through in reverse direction - 0.02
const float PL_DIFUSSE_ILLUMINATION_REDUCTION = 0.05f;           // how much of ambient light gets through in reverse direction - 0.05
const float PL_DIFUSSE_LIGHT_HALO_REDUCTION = 0.25f;             // intensity of light halo - 0.40
const float ATTNF = 0.3f;                                       // linear arrenuation factor 0.25
const float ATTNF2 = 0.8f;                                      // quadratic attenuation factor 0.75
const float HATTNF = 1.5f;                                       // light halo -- linear arrenuation factor - 1.5
const float HATTNF2 = 6.0f;                                      // light halo quadratic attenuation factor - 5.0
const float MAXLIGHT = 0.999f;                                   // max contribution to avoid overburning; - 0.999
const float IGNORED_ATTN_DISTANCE = 0.012f;                      // distance after attenuation starts taking effect - 0.012
const float ILLUMINATION_CUTOFF = 0.10f;                         // remove flickering, light FOV - 0.10
const float BEHIND_LIGHT_FACTOR = 0.02f;                         // ambient illumination behind light source - 0.02f
const float DISTANCE_LIGHT = 0.25f;                             // force illumination near the light source  - 0.475
const float LIGHT_POS_Y_OFFSET = 0.35f;                          // vertical light position change 
const float INTO_WALL = 0.01f;                                   // into wall target raycast offset: 0.01

out vec4 fragColor;                                              //300 es

// ----------------------------------------------------------------------------
// Function prototypes
// ----------------------------------------------------------------------------

vec3 CalcLight(vec3 lightPosition, vec3 FragPos, vec3 viewDir, vec3 normal, vec3 pointLightColor, float shininess, vec3 ambientColor, vec3 diffuseColor, vec3 specularColor, float ambientStrength, float diffuseStrength, float specularStrength, int inner, vec3 lightDirection);
bool Raycast(vec3 rayOrigin3D, vec3 rayTarget3D, float illumination);
vec2 worldToGridTexCoord(vec2 position2D);
vec3 worldToNormalizedTexCoord(vec2 position2D);
bool isOccluded(vec2 position2D);

bool Raycast3D(vec3 rayOrigin3D, vec3 rayTarget3D, float illumination);
bool isOccluded(vec3 position3D);
vec3 worldToNormalizedTexCoord3D(vec3 position3D);
bool isOmniDirectional(vec3 dir);

vec3 debugDisplay(bool occluded);
vec3 illuminationDisplay(float illumination);
vec3 occlusionDisplay(bool occluded);
vec3 RayDebug(vec3 rayOrigin3D, vec3 rayTarget3D, float illumination); 

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
    vec3 innerLight = CalcLight(uCameraPos, FragPos, viewDir, norm, innerLightColor, shininess, ambientColor, diffuseColor, specularColor, innerAmbientStrength, innerDiffuseStrength, innerSpecularStrength, 1, viewDir);

    // Sum point lights
    vec3 PL_output = vec3(0.0f);
    for (int i = 0; i < N_LIGHTS; i++) {
        if (uPointLights[i].x < 0.0f) {
            continue;
        }
        PL_output += CalcLight(uPointLights[i], FragPos, viewDir, norm, uLightColors[i], shininess, ambientColor, diffuseColor, specularColor, PL_AmbientStrength, PL_DiffuseStrength, PL_SpecularStrength, 0, uLightDirections[i]);
    }

    vec3 light = innerLight + PL_output;
    vec4 texelColor = texture(uSampler, vTextureCoord);
    if (texelColor.a < IGNORE_ALPHA) {
        discard;
    }

    fragColor = vec4(texelColor.rgb * light, texelColor.a);
}

vec3 CalcLight(vec3 lightPosition, vec3 FragPos, vec3 viewDir, vec3 normal, vec3 pointLightColor, float shininess, vec3 ambientColor, vec3 diffuseColor, vec3 specularColor, float ambientStrength, float diffuseStrength, float specularStrength, int inner, vec3 lightDirection) {
    lightPosition.y -= LIGHT_POS_Y_OFFSET;
    float lightPosDistance = distance(lightPosition, FragPos);
    vec3 lightDir = normalize(FragPos - lightPosition);
    vec3 directionOfOrthoLight = lightDirection;                                //it normal already!

    float invDistance = 1.0f / lightPosDistance;
    float attenuation = invDistance / (ATTNF + ATTNF2 * lightPosDistance);

    //is fragment illuminated by ligh source? omni dir is (128,128,128) so if x < 128.0 it is not omni dir, but directional!
    illumination = 1.0f;
    if (inner == 0 && !isOmniDirectional(lightDirection)) {
        illumination = max(dot(lightDir, normalize(lightDirection)), 0.0f);
    }

    vec3 ambientLight = vec3(0.0f);

    /* behind the light source, we don't care about occlusion! and we exlude 0-dir fireballs */
    if (inner == 0 && !isOmniDirectional(lightDirection) && dot(-lightDir, directionOfOrthoLight) > ILLUMINATION_CUTOFF) {
        ambientLight = pointLightColor * ambientStrength * attenuation * ambientColor * BEHIND_LIGHT_FACTOR;
        return ambientLight; // Shadow the fragment, it's behind the light
    }

    bool occluded = Raycast3D(lightPosition, FragPos, illumination);
    //bool occluded = Raycast(lightPosition, FragPos, illumination);

    //return debugDisplay(occluded);                                            //debug
    //return illuminationDisplay(illumination);                                 //debug
    //return occlusionDisplay(isOccluded(RayDebug(lightPosition, FragPos, illumination)));

    bool isLight = false;
    if (lightPosDistance < DISTANCE_LIGHT) {
        isLight = true;
    }

    //ambient

    if (inner == 1) {
        ambientLight = pointLightColor * ambientStrength * ambientColor;
    } else {
        ambientLight = pointLightColor * ambientStrength * attenuation * ambientColor;
    }

    // Diffuse lighting  
    float diffLight = max(dot(normal, lightDir), 0.0f);
    float diffView = max(dot(normal, viewDir), 0.0f);
    float diff = 0.95f * diffLight + 0.05f * diffView;
    vec3 diffuselight = pointLightColor * diff * diffuseStrength * attenuation * diffuseColor;

    // specular shading
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0f), shininess);
    vec3 specularLight = pointLightColor * spec * specularStrength * attenuation * specularColor;

//|| dot(-lightDir, directionOfOrthoLight) > BACK_ILLUMINATION_CUTOFF
    if (illumination < ILLUMINATION_CUTOFF) {
        if (isLight) {
            float invlightDistance = 1.0f / lightPosDistance;
            float attenuationHalo = invlightDistance / (HATTNF + HATTNF2 * lightPosDistance);
            diffuselight *= PL_DIFUSSE_LIGHT_HALO_REDUCTION * attenuationHalo;
        } else {
            diffuselight *= PL_DIFUSSE_ILLUMINATION_REDUCTION;
        }

        if (lightPosDistance > IGNORED_ATTN_DISTANCE) {
            ambientLight *= PL_AMBIENT_ILLUMINATION_REDUCTION;
        }
    } else if (occluded && inner == 0) {
        return PL_AMBIENT_OCCLUSION * ambientLight + PL_DIFFUSE_OCCLUSION * diffuselight;
    }

    return clamp(ambientLight + diffuselight + specularLight, 0.0f, MAXLIGHT);
}

bool Raycast3D(vec3 rayOrigin3D, vec3 rayTarget3D, float illumination) {
    vec3 direction = rayTarget3D - rayOrigin3D;
    vec3 step = sign(direction);
    float illumination2 = (illumination + EPSILON) * (illumination + EPSILON);
    vec3 cellTarget = floor(rayTarget3D - step * INTO_WALL * illumination2);            //pull it from the wall boundary so they are illuminated if on border

    vec3 tDelta = 1.0f / max(abs(direction), vec3(EPSILON));
    vec3 startCell = floor(rayOrigin3D);                                                // integer cell containing the origin - assured non boundary

    // how far to the next boundary
    vec3 tMax = vec3(0.0f);

    for (int i = 0; i < 3; i++) {
        float distToBoundary = (step[i] > 0.0f) ? (1.0f - fract(rayOrigin3D[i])) : fract(rayOrigin3D[i]);
        tMax[i] = distToBoundary * tDelta[i];
    }

    vec3 currentCell = rayOrigin3D;

    // *** Walk along the ray in 3D, up to MAX_STEPS ***
    for (int i = 0; i < MAX_STEPS; i++) {

        if (isOccluded(currentCell)) {
            return true;
        }

        if ((step.x > 0.0f && currentCell.x >= cellTarget.x) || (step.x < 0.0f && currentCell.x <= cellTarget.x)) {
            if ((step.y > 0.0f && currentCell.y >= cellTarget.y) || (step.y < 0.0f && currentCell.y <= cellTarget.y)) {
                if ((step.z > 0.0f && currentCell.z >= cellTarget.z) || (step.z < 0.0f && currentCell.z <= cellTarget.z)) {
                    return false; // Target reached, stop raycasting
                }
            }
        }

        if (floor(currentCell) == cellTarget) {
            return false;
        }

        // How far in t we must go to step in x vs y vs z
        if (tMax.x <= tMax.z) {  // Prioritize X and Z first
            if (tMax.x <= tMax.y) {
                tMax.x += tDelta.x;
                currentCell.x += step.x;
            } else {
                tMax.y += tDelta.y;
                currentCell.y += step.y;
            }
        } else {
            if (tMax.z <= tMax.y) {
                tMax.z += tDelta.z;
                currentCell.z += step.z;
            } else {
                tMax.y += tDelta.y;
                currentCell.y += step.y;
            }
        }

    }
    return false;                               // No occlusion detected, target not reached
}

bool Raycast(vec3 rayOrigin3D, vec3 rayTarget3D, float illumination) {
    //to XZ plane
    vec2 origin = rayOrigin3D.xz;
    vec2 target = rayTarget3D.xz;

    vec2 direction = target - origin;
    vec2 step = sign(target - origin);
    float illumination2 = (illumination + EPSILON) * (illumination + EPSILON);
    vec2 gridTarget = floor(target - step * INTO_WALL * illumination2);            // Adjusted target with directional offset so the target is reached when FragPOs is in the wall - iluminating wall   

    vec2 tDelta = abs(1.0f / max(abs(direction), vec2(EPSILON)));                            // How far to go in each direction to cross a grid line

    vec2 tMax;
    tMax.x = step.x > 0.0f ? (1.0f - fract(origin.x)) * tDelta.x : fract(origin.x) * tDelta.x;
    tMax.y = step.y > 0.0f ? (1.0f - fract(origin.y)) * tDelta.y : fract(origin.y) * tDelta.y;

    vec2 current = origin;

    for (int i = 0; i < MAX_STEPS; i++) {

        //if moved after below checks, removes flickering but lits back of the grid
        if (isOccluded(current)) {
            return true;
        }

        if (floor(current) == gridTarget) {
            return false;
        }
        if ((step.x > 0.0f && current.x >= gridTarget.x) || (step.x < 0.0f && current.x <= gridTarget.x)) {
            if ((step.y > 0.0f && current.y >= gridTarget.y) || (step.y < 0.0f && current.y <= gridTarget.y)) {
                return false;                                                                                        // Target reached
            }
        }

        if (tMax.x < tMax.y) {
            tMax.x += tDelta.x;
            current.x += step.x;
        } else {
            tMax.y += tDelta.y;
            current.y += step.y;
        }
    }
    return false;                               // No occlusion detected, target not reached
}

vec2 worldToGridTexCoord(vec2 position2D) {
    return vec2(floor(position2D.x), floor(position2D.y));
}

vec3 worldToNormalizedTexCoord(vec2 position2D) {
    return vec3(position2D.x / uGridSize.x, position2D.y / uGridSize.y, 0.0f);
}

bool isOccluded(vec2 position2D) {
    vec3 texCoord = worldToNormalizedTexCoord(position2D);
    float occlusion = texture(uOcclusionMap, texCoord).r; // Sample red channel
    return occlusion > 0.5f; //  >0.5 indicates impassable
}

bool isOccluded(vec3 position3D) {
    // Convert from world position to [0,1] texture coordinates
    vec3 texCoord = worldToNormalizedTexCoord3D(position3D);
    float occlusion = texture(uOcclusionMap, texCoord).r;
    return (occlusion >= 0.5f);
}

vec3 worldToNormalizedTexCoord3D(vec3 position3D) {
    //need to swap z and y
    return vec3(position3D.x / uGridSize.x, position3D.z / uGridSize.y, position3D.y / uGridSize.z);
}

bool isOmniDirectional(vec3 dir) {
    return length(dir) < 0.01f; // very small threshold
}

/*** DEBUG ***/
vec3 debugDisplay(bool occluded) {
    if (occluded) {
        return vec3(1.0f, 0.0f, 0.0f);
    } else
        return vec3(0.0f, 1.0f, 0.0f);
}

vec3 occlusionDisplay(bool occluded) {
    if (occluded) {
        return vec3(0.5f, 0.0f, 0.0f);
    } else {
        return vec3(0.0f, 0.1f, 0.0f);
    }
}

vec3 illuminationDisplay(float illumination) {
    return vec3(0.0f, illumination, 0.0f);
}

vec3 RayDebug(vec3 rayOrigin3D, vec3 rayTarget3D, float illumination) {
    vec3 direction = rayTarget3D - rayOrigin3D;
    float illumination2 = (illumination + EPSILON) * (illumination + EPSILON);
    vec3 adjustedTarget = rayTarget3D + normalize(direction) * INTO_WALL * illumination2;

    return adjustedTarget;
}