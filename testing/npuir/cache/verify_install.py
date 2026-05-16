# Quick verification that the new cache features are active
import tilelang
import tilelang.cache
from tilelang.cache.kernel_cache import KernelCache

# Check new features exist
assert hasattr(KernelCache, '_get_staging_root'), 'Missing _get_staging_root'
assert hasattr(KernelCache, '_safe_write_file'), 'Missing _safe_write_file'
assert hasattr(KernelCache, '_is_complete_cache_dir'), 'Missing _is_complete_cache_dir'
assert hasattr(KernelCache, '_get_cache_namespace'), 'Missing _get_cache_namespace'
assert hasattr(KernelCache, '_get_tilelang_lib_stamp'), 'Missing _get_tilelang_lib_stamp'

from tilelang.utils.language import get_prim_func_name
assert callable(get_prim_func_name)

print('CACHE_OPTIMIZATION_V2_ACTIVE')
print(f'tilelang version: {tilelang.__version__}')
print(f'Cache dir: {tilelang.cache.get_cache_dir()}')
print('VERIFIED: new process-safe cache features present')
