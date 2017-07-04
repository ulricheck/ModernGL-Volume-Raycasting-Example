#version 410

layout (location = 0) in vec3 VerPos;


out vec3 EntryPoint;
out vec4 ExitPointCoord;

uniform mat4 Mvp;

void main()
{
    EntryPoint = VerPos;
    gl_Position = Mvp * vec4(VerPos,1.0);
    ExitPointCoord = gl_Position;  
}