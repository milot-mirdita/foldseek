#ifndef PROST_T5_H
#define PROST_T5_H

#include <string>

struct llama_model;
struct llama_context;

class ProstT5 {
public:
    ProstT5(const std::string& model_file, bool gpu);
    ~ProstT5();
    
    std::string predict(const std::string& aa);
    void perf();

    llama_model * model;
    llama_context * ctx;
};


#endif // PROST_T5_H