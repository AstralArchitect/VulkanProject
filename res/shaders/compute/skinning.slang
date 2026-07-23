// Both structs use C-style arrays to match the binary layout of the GPU buffers
// uploaded from the CPU (see OutputVertex in renderer_advanced_types.h).
struct InputVertex {
    float p[3];
    float n[3];
    float uv[2];
    float t[4]; // tangent; t[3] is handedness (-1 or +1)
};

struct OutputVertex {
    float p[3];
    float n[3];
    float uv[2];
    float t[4];
};

// Joint data is stored in two separate buffers rather than being embedded in
// InputVertex, so the rest-pose geometry buffer never needs to change.
[[vk::binding(0, 0)]] StructuredBuffer<InputVertex>    input_vertices;
[[vk::binding(1, 0)]] RWStructuredBuffer<OutputVertex> output_vertices;
[[vk::binding(2, 0)]] StructuredBuffer<float4x4>       joint_matrices;
[[vk::binding(3, 0)]] StructuredBuffer<uint4>          joint_indices;
[[vk::binding(4, 0)]] StructuredBuffer<float4>         joint_weights;

// Push constants for per-dispatch data
struct SkinPushConstants {
    uint vertex_count;
};
[[vk::push_constant]] SkinPushConstants push_constants;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint vertex_id = dispatchThreadID.x;
    if (vertex_id >= push_constants.vertex_count) return;

    InputVertex v = input_vertices[vertex_id];
    uint4  j_idx  = joint_indices[vertex_id];
    float4 j_w    = joint_weights[vertex_id];

    // Build the blended skin matrix from the four joint influences.
    // This is the core of the Linear Blend Skinning (LBS) equation:
    // SkinMatrix = sum(weight_i * JointMatrix_i) for i in [0, 3]
    float4x4 skin_matrix =
        j_w.x * joint_matrices[j_idx.x] +
        j_w.y * joint_matrices[j_idx.y] +
        j_w.z * joint_matrices[j_idx.z] +
        j_w.w * joint_matrices[j_idx.w];

    float3 v_pos = float3(v.p[0], v.p[1], v.p[2]);
    float3 v_nrm = float3(v.n[0], v.n[1], v.n[2]);
    float4 v_tan = float4(v.t[0], v.t[1], v.t[2], v.t[3]);

    // Transform the position using the full 4x4 skin matrix
    float4 animated_pos = mul(skin_matrix, float4(v_pos, 1.0));

    // Normals and tangents use only the 3x3 rotational portion (no translation).
    float3x3 skin_rot = float3x3(
        skin_matrix[0].xyz,
        skin_matrix[1].xyz,
        skin_matrix[2].xyz);

    float3 animated_normal  = normalize(mul(skin_rot, v_nrm));
    float3 animated_tangent = normalize(mul(skin_rot, v_tan.xyz));

    // Write the animated data to the output buffer element-by-element
    // (required because OutputVertex uses C-style arrays, not vector types).
    OutputVertex out_v;
    out_v.p[0] = animated_pos.x;
    out_v.p[1] = animated_pos.y;
    out_v.p[2] = animated_pos.z;

    out_v.n[0] = animated_normal.x;
    out_v.n[1] = animated_normal.y;
    out_v.n[2] = animated_normal.z;

    out_v.t[0] = animated_tangent.x;
    out_v.t[1] = animated_tangent.y;
    out_v.t[2] = animated_tangent.z;
    out_v.t[3] = v_tan.w; // preserve handedness

    out_v.uv[0] = v.uv[0]; // UV coordinates are not affected by skinning
    out_v.uv[1] = v.uv[1];

    output_vertices[vertex_id] = out_v;
}