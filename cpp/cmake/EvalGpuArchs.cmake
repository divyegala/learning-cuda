function(evaluate_gpu_archs gpu_archs)
  set(eval_file ${PROJECT_BINARY_DIR}/eval_gpu_archs.cu)
  set(eval_exe ${PROJECT_BINARY_DIR}/eval_gpu_archs)
  file(WRITE ${eval_file}
    "
#include <cstdio>
#include <set>
#include <string>
using namespace std;
int main(int argc, char** argv) {
  set<string> archs;
  int nDevices;
  if((cudaGetDeviceCount(&nDevices) == cudaSuccess) && (nDevices > 0)) {
    for(int dev=0;dev<nDevices;++dev) {
      char buff[32];
      cudaDeviceProp prop;
      if(cudaGetDeviceProperties(&prop, dev) != cudaSuccess) continue;
      sprintf(buff, \"%d%d\", prop.major, prop.minor);
      archs.insert(buff);
    }
  }
  if(archs.empty()) {
    printf(\"ALL\");
  } else {
    bool first = true;
    for(set<string>::const_iterator itr=archs.begin();itr!=archs.end();++itr) {
      printf(first? \"%s\" : \";%s\", itr->c_str());
      first = false;
    }
  }
  printf(\"\\n\");
  return 0;
}
")
  execute_process(
    COMMAND ${CUDA_NVCC_EXECUTABLE}
      -o ${eval_exe}
      --run
      ${eval_file}
    OUTPUT_VARIABLE __gpu_archs
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  set(__gpu_archs_filtered "${__gpu_archs}")
  foreach(arch ${__gpu_archs})
    if (arch VERSION_LESS 60)
      list(REMOVE_ITEM __gpu_archs_filtered ${arch})
    endif()
  endforeach()
  if (NOT __gpu_archs_filtered)
    message(FATAL_ERROR "No supported GPU arch found on this system")
  endif()
  message("Auto detection of gpu-archs: ${__gpu_archs_filtered}")
  set(${gpu_archs} ${__gpu_archs_filtered} PARENT_SCOPE)
endfunction(evaluate_gpu_archs)