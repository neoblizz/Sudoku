// ----------------------------------------------------------------
// Sudoku -- Puzzle Solver on GPU using CUDA
// ----------------------------------------------------------------

/**
 * @file
 * error_utils.cu
 *
 * @brief Error handling utility routines
 */

 /**
  * Error checking for CUDA memory allocations.
  */
#define ERROR_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file,
                      int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"Cuda error in file '%s' in line '%d': %s\n",
              file, line, cudaGetErrorString(code));
      if (abort) exit(code);
   }
}

#define CUDA_SAFE_CALL_NO_SYNC(call) do {                               \
  cudaError err = call;                                                 \
  if( cudaSuccess != err) {                                             \
    fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",       \
                __FILE__, __LINE__, cudaGetErrorString( err) );         \
    exit(EXIT_FAILURE);                                                 \
    } } while (0)

#define CUDA_SAFE_CALL(call) do {                                       \
  CUDA_SAFE_CALL_NO_SYNC(call);                                         \
  cudaError err = cudaThreadSynchronize();                              \
  if( cudaSuccess != err) {                                             \
     fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",      \
                 __FILE__, __LINE__, cudaGetErrorString( err) );        \
     exit(EXIT_FAILURE);                                                \
     } } while (0)

/**
 * @brief prints an error if the usage of commandline is incorrect
 * @param[in] argv          List of arguments.
 */

 void usage(char ** argv) {
    fprintf(stderr, "Usage: %s n -{gfx,txt} filename.txt\n", argv[0]);
    exit(1);
 }

 void overFilename(char ** argv) {
    fprintf(stderr, "Overflow: Filename %s is too long.", argv[3]);
    exit(1);
 }
