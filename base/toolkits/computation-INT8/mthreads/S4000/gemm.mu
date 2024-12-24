/* Copyright @2020-2024 Moore Threads Technology Co., Ltd("Moore Threads"). All
 * rights reserved.
 *
 * This software ("this software and its documentations" or "the software") is
 * protected by Copyright and the information contained herein is confidential.
 *
 * The software contained herein is PROPRIETARY to Moore Threads and is being
 * provided under the terms and conditions of a form of Moore Threads software
 * license agreement by and between Moore Threads and Licensee ("License
 * Agreement") or electronically accepted by Licensee. Notwithstanding any
 * terms or conditions to the contrary in the License Agreement, copy or
 * disclosure of the software to any third party without the express written
 * consent of Moore Threads is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE LICENSE
 * AGREEMENT, MOORE THREADS MAKES NO REPRESENTATION ABOUT ANY WARRANTIES,
 * INCLUDING BUT NOT LIMITED TO THE SUITABILITY OF THE SOFTWARE FOR ANY
 * PURPOSE. IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF
 * ANY KIND. MOORE THREADS DISCLAIMS ALL WARRANTIES WITH REGARD TO THE
 * SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL MOORE THREADS BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THE SOFTWARE.
 */
#include <musa_runtime.h>
#include <musa_fp16.h>
#include <musa_bf16.h>

#include <chrono>
#include <map>
#include <thread>
#include <type_traits>
#include <typeinfo>

#include <iostream>
#include <mudnn.h>
#include <cstring>

using qint8 = int8_t;

#define SHOW printf

struct MatMulParam {
    bool split_k{ false };
    bool trans_a{ false };
    bool trans_b{ false };
    int batch{ 1 };
    int m{ 8192 };
    int n{ 8192 };
    int k{ 8192 };
    double alpha{ 1.0 };
    double beta{ 0.0 };
    double gamma{ 0.0 };
    int mode{ 0 }; // 0 tensor, 1 scalar
};

#define CHECK_MUSA(...)                                                        \
  do {                                                                         \
    int err = CheckMusaError(__VA_ARGS__, #__VA_ARGS__, __FILE__, __LINE__);   \
    if (err)                                                                   \
      exit(err);                                                               \
  } while (0)

#define CHECK_ERR(...)                                                         \
  do {                                                                         \
    int err = CheckError(__VA_ARGS__, #__VA_ARGS__, __FILE__, __LINE__);       \
    if (err)                                                                   \
      exit(err);                                                               \
  } while (0)

int CheckMusaError(musaError_t code, const char* expr, const char* file,
    int line) {
    if (code) {
        printf("MUSA error at %s:%d, code=%d (%s) in '%s'", file, line, (int)code,
            musaGetErrorString(code), expr);
        return 1;
    }
    return 0;
}

int CheckError(bool code, const char* expr, const char* file, int line) {
    if (code) {
        printf("General error at %s:%d, code=%d (%s) in '%s'", file, line,
            (int)code, "general error", expr);
        return 1;
    }
    return 0;
}

void MemFree(void* ptr) {
    if (ptr) {
        musaFree(ptr);
    }
}

::musa::dnn::MemoryHandler MemoryFunc(size_t size) {
    void* data = nullptr;
    if (size) {
        musaMalloc(&data, size);
        musaMemset(data, 0, size);
    }
    return ::musa::dnn::MemoryHandler(data, MemFree);
}

enum DType {
    f32,
    f16,
    q8,
    bf16,
};


class TestMatMul {
public:
    inline float F32MaskFormatTF32(float f) {
        unsigned int t = 0;
        std::memcpy(&t, &f, sizeof(f));
        // 1110 0000 0000 0000
        t = t & 0xffffe000;
        std::memcpy(&f, &t, sizeof(f));
        return f;
    }

    TestMatMul(const musaStream_t& _stream, const int _device_id, const DType _dtype, const MatMulParam _param, const int _iters)
    {
        stream = _stream;
        device_id = _device_id;
        dtype = _dtype;
        dtype_size = 4;

        switch (dtype) {
            case DType::f32:
                dtype_str = "float32";
                dtype_size = 4;
                break;
            case DType::f16:
                dtype_str = "float16";
                dtype_size = 2;
                break;
            case DType::bf16:
                dtype_str = "bfloat16";
                dtype_size = 2;
                break;
            case DType::q8:
                dtype_str = "qint8";
                dtype_size = 1;
                break;
            default:
                bool DType_Not_Suppoted = true;
                CHECK_ERR(DType_Not_Suppoted);
                break;
        }
        split_k = _param.split_k;
        trans_a = _param.trans_a;
        trans_b = _param.trans_b;
        batch = _param.batch;
        m = _param.m;
        n = _param.n;
        k = _param.k;
        alpha = _param.alpha;
        beta = _param.beta;
        gamma = _param.gamma;
        mode = _param.mode;

        iters = _iters;

        handle = new ::musa::dnn::Handle(device_id);
        handle->SetStream(stream);
    };
    ~TestMatMul() {
#define FREE_H(_PTR)                                                           \
  if (_PTR != nullptr) {                                                       \
    operator delete(_PTR);                                                     \
  }
#define FREE_D(_PTR)                                                           \
  if (_PTR != nullptr) {                                                       \
    CHECK_MUSA(musaFree(_PTR));                                                \
  }

        FREE_H(h_buf_a);
        FREE_H(h_buf_b);
        FREE_H(h_buf_c);
        FREE_H(h_buf_o);
        FREE_H(h_buf_z);

        FREE_D(d_a);
        FREE_D(d_b);
        FREE_D(d_c);
        FREE_D(d_z);

        FREE_D(d_base);
        FREE_D(d_bool);
        FREE_D(d_nonz);
        FREE_H(h_nonz);

#undef FREE_H
#undef FREE_D

        if (handle) {
            delete handle;
        }
    };

    bool Test() {
        // check parameters
        CheckParams();
        // initial memory && dnn tensor op
        Init();
        // warm up && prepare base golden
        Exec();
        // main loop
        float elapsed_ms = 0.f;
        musaEvent_t start, stop;
        if (performance) {
            CHECK_MUSA(musaEventCreate(&start));
            CHECK_MUSA(musaEventCreate(&stop));
            CHECK_MUSA(musaEventRecord(start, stream));
        }

        std::chrono::milliseconds bubble_time(bubble);
        std::chrono::milliseconds duration_time(duration);
        std::chrono::milliseconds show_gap_time(60000);
        int show_gap_count = 0;
        auto start_time = std::chrono::steady_clock::now();
        auto current_time = start_time;
        const bool blocking = (bubble > 0) || (iters == 0 && duration > 0);
        int stable_check_gap_count = 1;
        int run_iters_count = 0;
        int i = 0;
        while ((iters > 0 && i < iters) ||
            (iters == 0 && (current_time - start_time) <= duration_time)) {
            // operator running
            Exec(blocking);

            if (bubble > 0) {
                // SHOW("sleeping %d ms\n", bubble);
                std::this_thread::sleep_for(bubble_time);
            }
            current_time = std::chrono::steady_clock::now();
            if ((iters == 0 && duration > 0) &&
                (current_time - start_time) > show_gap_time * show_gap_count) {
                std::cout << "--- now execution time passed "
                    << (show_gap_time * show_gap_count).count() << std::endl;
                show_gap_count++;
            }
            // SHOW("run loop %d\n", run_iters_count);
            i++, stable_check_gap_count++, run_iters_count++;
        }
        // performance testing and stability checking are mutually exclusive
        if (performance) {
            CHECK_MUSA(musaEventRecord(stop, stream));
            CHECK_MUSA(musaEventSynchronize(stop));
            CHECK_MUSA(musaEventElapsedTime(&elapsed_ms, start, stop));
            elapsed_ms = elapsed_ms / run_iters_count;
            ShowPerformance(elapsed_ms, (size_t)m * n * k * 2 / elapsed_ms * 1e-6,
                !stable_check);
            CHECK_MUSA(musaEventDestroy(start));
            CHECK_MUSA(musaEventDestroy(stop));
        }
        return true;
    }

    void ShowPerformance(float t, float gops, bool credible) {
        // SHOW("dev_time : %f, gops : %f %s\n", t, credible ? gops : 0.f,
        //     credible
        //     ? " "
        //     : " - the performance is not credible when enable stable checking");
        SHOW("Average INT8 Single Op Duration:%f us\n", t * 1.0e3);
        SHOW("[FlagPerf Result]computation-INT8=%f TOPS\n", gops / 1.0e3);

    }

private:
    void* h_buf_a = nullptr;
    void* h_buf_b = nullptr;
    void* h_buf_c = nullptr;
    void* h_buf_o = nullptr;
    void* h_buf_z = nullptr;

    void* d_a = nullptr;
    void* d_b = nullptr;
    void* d_c = nullptr;
    void* d_z = nullptr;

    void* d_base = nullptr;
    void* d_bool = nullptr;
    void* d_nonz = nullptr;
    int64_t* h_nonz = nullptr;

    bool result_check = false;
    bool stable_check = false;
    bool stable_check_gpu = false;
    bool performance = true;
    bool verbose = false;
    int iters = 1;
    int duration = 0;
    int bubble = 0;
    int gap = 1;
    uint seed = 2333;

    DType dtype = DType::f32;
    std::string dtype_str = "float32";
    size_t dtype_size = 4;
    bool split_k = false;
    bool trans_a = false;
    bool trans_b = false;
    int batch = 1;
    int m = 1;
    int n = 1;
    int k = 1;
    double alpha = 1.0;
    double beta = 0.0;
    double gamma = 0.0;
    int mode = 0;

    // qint8 variables
    const float scale_a = 1.f / 32.f;
    const float scale_b = 1.f / 32.f;
    const float scale_c = 32.f;

    // mudnn variables
    musaStream_t stream;
    int device_id;
    ::musa::dnn::Handle* handle;
    ::musa::dnn::MatMul op;

    ::musa::dnn::Tensor tensor_a;
    ::musa::dnn::Tensor tensor_b;
    ::musa::dnn::Tensor tensor_c;
    ::musa::dnn::Tensor tensor_z;
    ::musa::dnn::Tensor tensor_base;
    ::musa::dnn::Tensor tensor_bool;
    ::musa::dnn::Tensor tensor_nonz;

private:


    ::musa::dnn::Tensor::Type GetmuDNNType(const std::string& dtype) {
        using T = ::musa::dnn::Tensor::Type;
        static std::map<std::string, T> type_mapping = {
            {"int8", T::INT8},
            {"int16", T::INT16},
            {"int32", T::INT32},

            {"int", T::INT64},
            {"int64", T::INT64},

            {"uint8", T::UINT8},
            {"uint16", T::UINT16},
            {"uint32", T::UINT32},

            {"uint", T::UINT64},
            {"uint64", T::UINT64},

            {"half", T::HALF},
            {"float16", T::HALF},
            {"bfloat16", T::BFLOAT16},

            {"float32", T::FLOAT},
            {"qint8", T::QINT8},

            {"float", T::FLOAT},
            {"float64", T::DOUBLE},
            {"double", T::DOUBLE},

            {"bool", T::BOOL},
        };
        if (type_mapping.find(dtype) != type_mapping.end()) {
            return type_mapping.at(dtype);
        }
        else {
            std::cerr << "GetmuDNNType error : " << dtype << std::endl;
            return type_mapping.at("float");
        }
    }
    bool CheckParams() {
        bool pass = true;
        // param checking
        if (mode != 0 && mode != 1) {
            std::cerr << "MatMul mode setting error, fallback 0" << std::endl;
            mode = 0;
        }
        if (m <= 0 || n <= 0 || k <= 0) {
            std::cerr << "MatMul param setting error, fallback 1" << std::endl;
            m = m > 0 ? m : 1;
            n = n > 0 ? n : 1;
            k = k > 0 ? k : 1;
        }
        if (gamma != 0) {
            std::cerr << "MatMul unsupported gamma != 0 temporarily, fallback 0"
                << std::endl;
            gamma = 0;
        }
        if (beta != 0) {
            if (mode == 0) {
                std::cerr << "MatMul unsupported beta != 0 when mode == 0, fallback 0"
                    << std::endl;
                beta = 0;
            }

        }
        if (dtype == DType::q8) {
            // To be removed when binary supports QINT8
            if (stable_check_gpu) {
                std::cerr
                    << "MatMul unsupported qint8 for stable_check_gpu, fallback cpu "
                    << std::endl;
                stable_check_gpu = false;
            }
            if (mode != 0) {
                std::cerr << "MatMul mode must be 0 when qint8, fallback 0"
                    << std::endl;
                mode = 0;
            }
        }

        // #define BOOL(_VAL) _VAL ? "√" : "×"
        //         SHOW("MatMul Param  : m %d, n %d, k %d, transpose %c%c, type %s, mode %d, "
        //             "alpha %.2f, beta %.2f, gamma %.2f\n",
        //             m, n, k, trans_a ? 'T' : 'N', trans_b ? 'T' : 'N', dtype_str.c_str(),
        //             mode, alpha, beta, gamma);
        //         SHOW("MatMul Option : iters %d, duration %d, bubble %d, stable_check_gap "
        //             "%d, seed %u, stable_check %s, stable_check_mode %s, result_check %s, "
        //             "performance %s, verbose %s,\n",
        //             iters, duration, bubble, gap, seed, BOOL(stable_check),
        //             stable_check ? (stable_check_gpu ? "gpu" : "cpu") : "none",
        //             BOOL(result_check), BOOL(performance), BOOL(verbose));
        // #undef BOOL
        return pass;
    }

    bool Init() {
        size_t nr_elem_a = (size_t)(m)*k;
        size_t nr_elem_b = (size_t)(k)*n;
        size_t nr_elem_c = (size_t)(m)*n;
        size_t nr_elem_z = (size_t)(n);

        size_t size_a = nr_elem_a * dtype_size;
        size_t size_b = nr_elem_b * dtype_size;
        size_t size_c = nr_elem_c * dtype_size;
        size_t size_z = nr_elem_z * dtype_size;

        size_t mem_total, mem_free;
        CHECK_MUSA(musaMemGetInfo(&mem_free, &mem_total));
        size_t available_gpu_mem = mem_free;
        size_t total_gpu_mem = mem_total;
        size_t need_gpu_mem = size_a + size_b + size_c;
        if (gamma != 0) {
            need_gpu_mem += size_z;
        }
        if (stable_check && stable_check_gpu) {
            need_gpu_mem +=
                size_c + sizeof(bool) * nr_elem_c + sizeof(int64_t) * m * n * 2;
        }
        if ((need_gpu_mem > available_gpu_mem) || verbose) {
            SHOW("%s : Need Device Memory %.2f GiB, Available Device Memory %.2f GiB "
                "(Total %.2f GiB)\n",
                (need_gpu_mem > available_gpu_mem) ? "Error" : "Verbose",
                need_gpu_mem / 1024.f / 1024 / 1024,
                available_gpu_mem / 1024.f / 1024 / 1024,
                total_gpu_mem / 1024.f / 1024 / 1024);
        }
        CHECK_ERR(need_gpu_mem > available_gpu_mem);

        // host buffer
        h_buf_a = operator new(size_a); // new char[size_a]();
        h_buf_b = operator new(size_b); // new char[size_b]();
        h_buf_c = operator new(size_c); // new char[size_c]();
        h_buf_o = operator new(size_c); // new char[size_c]();

        // host data initialization
        if (dtype == DType::f16) {
            __half* d_a, * d_b, * d_c;
            std::vector<__half> h_buf_a(nr_elem_a, __float2half(1.0f));
            std::vector<__half> h_buf_b(nr_elem_b, __float2half(1.0f));
            std::vector<__half> h_buf_c(nr_elem_c, __float2half(1.0f));
        }
        else if (dtype == DType::bf16) {
            __mt_bfloat16* d_a, * d_b, * d_c;
            std::vector<__mt_bfloat16> h_buf_a(nr_elem_a, __float2bfloat16(1.0f));
            std::vector<__mt_bfloat16> h_buf_b(nr_elem_b, __float2bfloat16(1.0f));
            std::vector<__mt_bfloat16> h_buf_c(nr_elem_c, __float2bfloat16(1.0f));
        }
        else if (dtype == DType::q8) {
            qint8* d_a, * d_b, * d_c;
            std::vector<qint8> h_buf_a(nr_elem_a, qint8(1));
            std::vector<qint8> h_buf_b(nr_elem_b, qint8(1));
            std::vector<qint8> h_buf_c(nr_elem_c, qint8(1));
        }
        else {
            float* d_a, * d_b, * d_c;
            std::vector<float> h_buf_a(nr_elem_a, 1.0f);
            std::vector<float> h_buf_b(nr_elem_b, 1.0f);
            std::vector<float> h_buf_c(nr_elem_c, 1.0f);
        }

        // tensor float 32 format
        if ((dtype == DType::f32) && mode == 0) {
            for (size_t i = 0; i < nr_elem_a; i++) {
                ((float*)h_buf_a)[i] = (float)F32MaskFormatTF32(((float*)h_buf_a)[i]);
            }
            for (size_t i = 0; i < nr_elem_b; i++) {
                ((float*)h_buf_b)[i] = (float)F32MaskFormatTF32(((float*)h_buf_b)[i]);
            }
            for (size_t i = 0; i < nr_elem_c; i++) {
                ((float*)h_buf_c)[i] = (float)F32MaskFormatTF32(((float*)h_buf_c)[i]);
            }
        }

        // device buffer
        CHECK_MUSA(musaMalloc(&d_a, size_a));
        CHECK_MUSA(musaMalloc(&d_b, size_b));
        CHECK_MUSA(musaMalloc(&d_c, size_c));

        // transfer host data to device

        CHECK_MUSA(musaMemcpy(d_a, h_buf_a, size_a, musaMemcpyHostToDevice));
        CHECK_MUSA(musaMemcpy(d_b, h_buf_b, size_b, musaMemcpyHostToDevice));
        CHECK_MUSA(musaMemcpy(d_c, h_buf_c, size_c, musaMemcpyHostToDevice));

        // host and device buffer for gamma 
        if (gamma != 0) {
            h_buf_z = new char[size_z]();
            CHECK_MUSA(musaMalloc(&d_z, size_z));
            CHECK_MUSA(musaMemcpy(d_z, h_buf_z, size_z, musaMemcpyHostToDevice));
            if (dtype == DType::f16) {
                __half* d_z;
                std::vector<__half> h_buf_z(nr_elem_z, __float2half(1.0f));
            }
            else if (dtype == DType::bf16) {
                __mt_bfloat16* d_z;
                std::vector<__mt_bfloat16> h_buf_z(nr_elem_z, __float2bfloat16(1.0f));
            }
            else if (dtype == DType::q8) {
                qint8* d_z;
                std::vector<qint8> h_buf_z(nr_elem_z, qint8(1));
            }
            else {
                float* d_z;
                std::vector<float> h_buf_z(nr_elem_z, 1.0f);
            }
        }


        ::musa::dnn::Tensor::Type ttype = GetmuDNNType(dtype_str);
        tensor_a.SetAddr(d_a);
        tensor_a.SetType(ttype);
        if (DType::q8 == dtype) {
            tensor_a.SetQuantizationInfo({ scale_a }, { 0 });
        }
        if (trans_a) {
            tensor_a.SetNdInfo({ k, m });
        }
        else {
            tensor_a.SetNdInfo({ m, k });
        }

        tensor_b.SetAddr(d_b);
        tensor_b.SetType(ttype);
        if (DType::q8 == dtype) {
            tensor_b.SetQuantizationInfo({ scale_b }, { 0 });
        }
        if (trans_b) {
            tensor_b.SetNdInfo({ n, k });
        }
        else {
            tensor_b.SetNdInfo({ k, n });
        }

        tensor_c.SetAddr(d_c);
        tensor_c.SetType(ttype);
        tensor_c.SetNdInfo({ m, n });
        if (DType::q8 == dtype) {
            tensor_c.SetQuantizationInfo({ scale_c }, { 0 });
        }

        tensor_z.SetAddr(d_z);
        tensor_z.SetType(ttype);
        tensor_z.SetNdInfo({ n });

        CHECK_MUSA(musaStreamSynchronize(stream));
        CHECK_MUSA(musaDeviceSynchronize());


        op.SetTranspose(trans_a, trans_b);
        op.SetSplitK(split_k);
        op.SetAlpha(alpha);
        op.SetBeta(beta);
        op.SetGamma(gamma);
        op.SetComputeMode(static_cast<::musa::dnn::MatMul::ComputeMode>(mode));

        return true;
    }

    void Exec(bool sync = false) {
        CHECK_ERR(::musa::dnn::Status::SUCCESS !=
            op.RunWithBiasAdd(*handle, tensor_c, tensor_a, tensor_b, tensor_z, MemoryFunc));
        CHECK_MUSA(musaGetLastError());
        if (sync) {
            CHECK_MUSA(musaStreamSynchronize(stream));
        }
    }
};

int RunMatMul() {


    int device_id = 0;
    CHECK_MUSA(musaGetDevice(&device_id));

    MatMulParam param;
    const int iters = 50000;
    musaStream_t stream;
    CHECK_MUSA(musaStreamCreate(&stream));
    TestMatMul test_mm(stream, device_id, DType::q8, param, iters);
    bool ret = test_mm.Test();
    CHECK_MUSA(musaStreamDestroy(stream));
    return ret;
}


int main() {
    RunMatMul();
}