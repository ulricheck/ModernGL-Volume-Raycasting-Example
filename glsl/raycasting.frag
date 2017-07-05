#version 410

// These variables come from the previous vertex stage
in vec3 EntryPoint;                 // Entry point as a 3d texture coordinate
in vec4 ExitPointCoord;             // Entry point in world space

// Uniforms are variables that are passed in and modified from the host
uniform sampler2D ExitPoints;       // Exit points 
uniform sampler3D VolumeTex;        // The volumetric texture to be rendered
uniform sampler2D TransferFunc;     // The transfer function, discretized into a 2d texture
uniform float     StepSize;         // Step size of the samples, in relation to the texture coordinates
uniform vec2      ScreenSize;       // 
uniform vec4      BackgroundColor = vec4(1.0); // The background color

// The output color that is rendered onto the screen
layout (location = 0) out vec4 FragColor;

void main()
{
    // Sample the exit point from the texture of the previous pass 
    vec3 exitPoint = texture(ExitPoints, gl_FragCoord.st/ScreenSize).xyz;

    //background needs no raycasting
    if (EntryPoint == exitPoint) 
        discard;

    // ray setup
    vec3 dir = exitPoint - EntryPoint;    // compute the direction vector
    float len = length(dir);              // the length from front to back is calculated and used to terminate the ray
    float effectiveStepSize = StepSize*0.5; // the effective step size
    vec3 deltaDir = normalize(dir) * effectiveStepSize; // distance and direction between samples
    vec3 voxelCoord = EntryPoint;         // current sample position. We start at the entry point and advance throughoug the raycasting loop
    vec4 colorAccum = vec4(0.0);          // The accumulated color of the ray
    
    // the number of samples that will be taken along the ray
    int numSteps = int(ceil(len / effectiveStepSize));

    // This loops over all samples along the ray and integrates the color
    for(int i = 0; i < numSteps; i++)
    {
        // sample scalar intensity value from the volume
        float intensity = texture(VolumeTex, voxelCoord).x;

        // evaluate transfer function by sampling the texture
        vec4 colorSample = texture(TransferFunc, vec2(intensity, 0.));
        
        // We only perform blending if the sample is not fully transparent
        if (colorSample.a > 0.0) {
            // accomodate for variable sampling rates
            colorSample.a = 1.0 - pow(1.0 - colorSample.a, effectiveStepSize*200.0f);
            // front-to-back blending of the samples
            colorAccum.rgb += (1.0 - colorAccum.a) * colorSample.rgb * colorSample.a;
            colorAccum.a += (1.0 - colorAccum.a) * colorSample.a;
        }

        // advance the ray sample position
        voxelCoord += deltaDir;
        
        // if the opacity is (almost) saturated, we can stop raycasting
        if (colorAccum.a > .97) {
            colorAccum.a = 1.0;
            break;
        }
    }
    // blend the background color using an "under" blend operation
    FragColor = mix(BackgroundColor, colorAccum, colorAccum.a); 
    
    // Visualize the number of raycasting steps
    //FragColor = vec4(clamp(float(numSteps)/1500, 0, 1));

    // for testing
    //FragColor = vec4(EntryPoint, 1.0);
    //FragColor = vec4(exitPoint, 1.0);
    
    // Visualize the transfer function
    //FragColor = texture(TransferFunc, vec2(gl_FragCoord.s/ScreenSize.x, 0.));
    //FragColor *= FragColor.a;
}