// DXGI video memory budget query for the AMD adapter (local + non-local segments).
#if defined(_WIN32)
#include <windows.h>
#include <dxgi1_4.h>

extern "C" int gufo_dxgi_free_bytes(unsigned long long* free_bytes) {
  IDXGIFactory4* factory = nullptr;
  if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory4), reinterpret_cast<void**>(&factory))))
    return -1;
  int result = -1;
  IDXGIAdapter1* adapter = nullptr;
  for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
    DXGI_ADAPTER_DESC1 desc{};
    adapter->GetDesc1(&desc);
    IDXGIAdapter3* adapter3 = nullptr;
    if (desc.VendorId == 0x1002 && !(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) &&
        SUCCEEDED(adapter->QueryInterface(__uuidof(IDXGIAdapter3), reinterpret_cast<void**>(&adapter3)))) {
      DXGI_QUERY_VIDEO_MEMORY_INFO local{}, shared{};
      if (SUCCEEDED(adapter3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &local)) &&
          SUCCEEDED(adapter3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &shared))) {
        const auto avail = [](const DXGI_QUERY_VIDEO_MEMORY_INFO& m) {
          return m.Budget > m.CurrentUsage ? m.Budget - m.CurrentUsage : 0ull;
        };
        *free_bytes = avail(local) + avail(shared);
        result = 0;
      }
      adapter3->Release();
    }
    adapter->Release();
    if (result == 0)
      break;
  }
  factory->Release();
  return result;
}
#endif
