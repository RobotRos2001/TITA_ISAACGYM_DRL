import onnxruntime as ort
import numpy as np
import time

def load_onnx_model(onnx_path, num_threads=1, use_gpu=False):
    """加载 ONNX 模型，支持多线程和 GPU"""
    sess_options = ort.SessionOptions()

    # 设置多线程选项
    sess_options.intra_op_num_threads = num_threads
    sess_options.inter_op_num_threads = num_threads

    # 设置 ONNX 的执行提供者
    providers = ['CUDAExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
    
    # 加载模型
    session = ort.InferenceSession(onnx_path, sess_options=sess_options, providers=providers)
    return session

def run_inference(session, input_data, num_runs=100000):
    """运行推理并测量时间"""
    if not isinstance(input_data, np.ndarray):
        input_data = np.array(input_data)
    if input_data.shape != (1, 27):
        raise ValueError("Input data must have shape (1, 27)")

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # 记录推理时间
    start_time = time.time()

    for _ in range(num_runs):
        output = session.run([output_name], {input_name: input_data})

    end_time = time.time()

    avg_inference_time = (end_time - start_time) / num_runs
    return output[0], avg_inference_time

if __name__ == "__main__":
    # 设置 ONNX 模型路径
    onnx_path = "/home/server/isaacgym_envs/pointfoot-legged-gym/logs/tita_pointfoot_rough/exported/policies/policy.onnx"

    # 生成示例输入 (1, 27)
    input_data = np.random.randn(1, 27).astype(np.float32)

    # 设置多线程数（例如 28 线程）
    num_threads = 28

    # 加载并测试多线程 CPU 推理时间
    cpu_session = load_onnx_model(onnx_path, num_threads=num_threads, use_gpu=False)
    output_cpu, avg_time_cpu = run_inference(cpu_session, input_data)

    # 输出多线程 CPU 平均推理时间（毫秒）
    print("多线程 CPU 推理输出:", output_cpu)
    print(f"多线程 CPU 平均推理时间: {avg_time_cpu * 1000:.4f} ms")