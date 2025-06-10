#version 300 es
///vShader///
/*
* v1.1 Haunting the Hauntessa
*/
#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif

in vec4 aVertexPosition;
in vec3 aVertexNormal;
in vec2 aTextureCoord;

uniform mat4 uModelViewMatrix;
uniform mat4 uProjectionMatrix;
uniform mat4 uScale;
uniform mat4 uTranslate;
uniform mat4 uRotateY;

out vec2 vTextureCoord;
out vec3 FragPos;
out vec3 v_normal;

void main(void) {
    gl_Position = uProjectionMatrix * uModelViewMatrix * uTranslate * uRotateY * uScale * aVertexPosition;
    vTextureCoord = aTextureCoord;
    FragPos = vec3(aVertexPosition);
    vec4 transformedNormal = uRotateY * vec4(aVertexNormal, 0.0);
    v_normal = transformedNormal.xyz;
}