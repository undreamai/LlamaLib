
#ifdef USE_RUNTIME_DETECTION
#include "LLM_runtime.h" ///< Dynamic library loading and runtime management
#else
#include "LLM_service.h" ///< LLM service implementation
#endif

class UNDREAMAI_API LLMServiceBuilder {
private:
    std::string model_path_;
    int num_slots_ = 1;
    int num_threads_ = -1;
    int num_GPU_layers_ = 0;
    bool flash_attention_ = false;
    int context_size_ = 4096;
    int batch_size_ = 2048;
    bool embedding_only_ = false;
    std::vector<std::string> lora_paths_ = {};

public:
    LLMServiceBuilder& model(const std::string& path) { 
        model_path_ = path; 
        return *this; 
    }
    
    LLMServiceBuilder& numSlots(int val) { 
        num_slots_ = val; 
        return *this; 
    }
    
    LLMServiceBuilder& numThreads(int val) { 
        num_threads_ = val; 
        return *this; 
    }
    
    LLMServiceBuilder& numGPULayers(int val) { 
        num_GPU_layers_ = val; 
        return *this; 
    }
    
    LLMServiceBuilder& flashAttention(bool val) { 
        flash_attention_ = val; 
        return *this; 
    }
    
    LLMServiceBuilder& contextSize(int val) { 
        context_size_ = val; 
        return *this; 
    }
    
    LLMServiceBuilder& batchSize(int val) { 
        batch_size_ = val; 
        return *this; 
    }
    
    LLMServiceBuilder& embeddingOnly(bool val) { 
        embedding_only_ = val; 
        return *this; 
    }
    
    LLMServiceBuilder& loraPaths(const std::vector<std::string>& paths) { 
        lora_paths_ = paths; 
        return *this; 
    }
    
    LLMService* build() {
        LLMService* service = new LLMService(
            model_path_, 
            num_slots_, 
            num_threads_, 
            num_GPU_layers_, 
            flash_attention_, 
            context_size_, 
            batch_size_, 
            embedding_only_, 
            lora_paths_
        );
        return service;
    }
};
