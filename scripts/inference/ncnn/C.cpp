#include <opencv2/opencv.hpp>
#include <net.h>
#include <iostream>
#include <chrono>
#include <cmath>

ncnn::Net load_ncnn_model(const std::string& param_path, const std::string& bin_path) {
    ncnn::Net net;
    net.load_param(param_path.c_str());
    net.load_model(bin_path.c_str());
    return net;
}

ncnn::Mat preprocess_image(const std::string& image_path) {
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        throw std::runtime_error("Không thể load ảnh: " + image_path);
    }

    cv::resize(img, img, cv::Size(112, 112));
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    img.convertTo(img, CV_32FC3, 1.0 / 127.5, -1.0); // normalize

    // Convert HWC -> CHW
    std::vector<cv::Mat> channels(3);
    cv::split(img, channels);

    ncnn::Mat in(112, 112, 3);
    float* ptr = (float*)in;
    for (int c = 0; c < 3; c++) {
        memcpy(ptr + 112 * 112 * c, channels[c].data, 112 * 112 * sizeof(float));
    }

    return in;
}

std::vector<float> extract_ncnn_features(ncnn::Net& net, const std::string& image_path) {
    ncnn::Mat in_blob = preprocess_image(image_path);
    ncnn::Extractor ex = net.create_extractor();
    ex.input("input.1", in_blob);

    auto start = std::chrono::high_resolution_clock::now();
    ncnn::Mat out;
    ex.extract("683", out);
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "[NCNN] Thời gian chạy: " << ms << " ms" << std::endl;

    std::vector<float> features(out.w);
    for (int i = 0; i < out.w; ++i) {
        features[i] = out[i];
    }

    return features;
}

float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
}

void compare_faces_ncnn(const std::string& image1, const std::string& image2, ncnn::Net& net) {
    auto feat1 = extract_ncnn_features(net, image1);
    auto feat2 = extract_ncnn_features(net, image2);
    float sim = cosine_similarity(feat1, feat2);

    std::cout << "\n[INFO] Feature 1 (NCNN): ";
    for (int i = 0; i < 5; ++i) std::cout << feat1[i] << " ";
    std::cout << "\n\nMức độ tương đồng (NCNN): " << sim << std::endl;
}

int main() {
    std::string param_path = "w600k_r50.param";
    std::string bin_path = "w600k_r50.bin";
    std::string img1 = "dataset/train/Akshay Kumar/anh_11.jpg";
    std::string img2 = "dataset/train/Akshay Kumar/anh_27.jpg";

    try {
        auto net = load_ncnn_model(param_path, bin_path);
        compare_faces_ncnn(img1, img2, net);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << std::endl;
    }

    return 0;
}
