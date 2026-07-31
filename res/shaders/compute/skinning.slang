// Both structs use C-style arrays to match the binary layout of the GPU buffers
// uploaded from the CPU (see OutputVertex in renderer_advanced_types.h).
struct InputVertex {
    float4 pos;      // pos.xyz, padding
    float4 normal;   // normal.xyz, padding
    float4 uv;       // uv.xy, padding
    uint4 joints;
    float4 weights;
};

struct OutputVertex {
    float4 pos;
    float4 normal;
    float4 uv;
    uint4 joints;
    float4 weights;
};

[[vk::binding(0, 0)]] StructuredBuffer<InputVertex>    input_vertices;
[[vk::binding(1, 0)]] RWStructuredBuffer<OutputVertex> output_vertices;
[[vk::binding(2, 0)]] StructuredBuffer<column_major float4x4>       joint_matrices;


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

    if (vertex_id >= push_constants.vertex_count) {
        return;
    }

    InputVertex v = input_vertices[vertex_id];
    uint4  j_idx  = v.joints;
    float4 j_w    = v.weights;

    // Build the blended skin matrix from the four joint influences.
    // This is the core of the Linear Blend Skinning (LBS) equation:
    // SkinMatrix = sum(weight_i * JointMatrix_i) for i in [0, 3]
    float4x4 skin_matrix =
        j_w.x * joint_matrices[j_idx.x] +
        j_w.y * joint_matrices[j_idx.y] +
        j_w.z * joint_matrices[j_idx.z] +
        j_w.w * joint_matrices[j_idx.w];

    float weight_sum = j_w.x + j_w.y + j_w.z + j_w.w;
    if (weight_sum < 0.01f) {
        skin_matrix = float4x4(
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1
        );
    }

    // Apply skinning matrix to position (assuming w=1.0 for positions)
    float4 local_pos = float4(v.pos.xyz, 1.0f);
    float4 skinned_pos = mul(skin_matrix, local_pos);

    // Apply skinning matrix to normal (assuming w=0.0 for directions)
    // NOTE: In a rigorous engine, we would use inverse-transpose of the 3x3 block,
    // but standard LBS often just uses the same matrix (which works for rigid transforms).
    float4 local_normal = float4(v.normal.xyz, 0.0f);
    float4 skinned_normal = mul(skin_matrix, local_normal);

    OutputVertex out_v;
    out_v.pos = float4(skinned_pos.xyz, v.pos.w);
    out_v.normal = float4(normalize(skinned_normal.xyz), v.normal.w);
    out_v.uv = v.uv;
    out_v.joints = v.joints;
    out_v.weights = v.weights;

    output_vertices[vertex_id] = out_v;
}