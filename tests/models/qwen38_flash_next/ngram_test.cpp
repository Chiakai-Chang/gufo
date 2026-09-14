// PLE n-gram hashing and disk row reads, checked without the model.
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

#include "src/models/qwen38_flash_next/ngram.hpp"

namespace q = gufo::models::qwen38_flash_next;

namespace {

q::Config FlashNextPle() {
  q::Config c;
  c.ple_layer = 1;
  c.ple_ngram_size = 3;
  c.ple_heads_per_ngram = 8;
  c.ple_heads = 16;
  c.ple_head_dim = 160;
  c.ple_conv_kernel = 4;
  c.ple_eos_token = 248044;
  c.ple_multipliers = {23703573157769ULL, 20109073645365ULL, 8052911324071ULL};
  c.ple_head_offsets = {0,         20000003,  40000026,  60000059,
                        80000106,  100000165, 120000228, 140000297,
                        160000374, 180000455, 200000548, 220000655,
                        240000802, 260000955, 280001114, 300001275};
  c.ple_head_vocab = {20000003, 20000023, 20000033, 20000047,
                      20000059, 20000063, 20000069, 20000077,
                      20000081, 20000093, 20000107, 20000147,
                      20000153, 20000159, 20000161, 20000171};
  c.ple_rows = 320001446;
  return c;
}

int failures = 0;

void Check(bool ok, const char* what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
  }
}

// Expected rows computed independently (Python, llama.cpp hash) for
// [<|im_start|>, "user", "\n", EOS, "Hello", ","]. The EOS at position 3
// cuts the window: position 4 hashes (9707, EOS, EOS), position 5
// (11, 9707, EOS).
void TestHash() {
  const q::Config c = FlashNextPle();
  const std::array<std::int32_t, 6> tokens{151644, 872, 198, 248044, 9707, 11};
  const std::array<std::array<std::uint32_t, 16>, 6> expected{{
      {9213577, 25503567, 59963735, 67280845, 97263283, 108604657, 126879742,
       153604170, 163991649, 192963728, 210734887, 224850479, 244505954,
       265606139, 292960741, 312140635},
      {6954764, 37428000, 54939688, 74004108, 95568253, 116575022, 138540160,
       142009604, 165349151, 193458441, 202840010, 231812024, 252625381,
       274082424, 294711253, 318927564},
      {6746895, 31942778, 54550641, 78212738, 81361934, 102413837, 123993646,
       146103739, 175246506, 190873805, 206877969, 222022136, 242132242,
       262461054, 282619366, 303774792},
      {16849591, 37537304, 42290399, 73883125, 88405327, 114186760, 123740665,
       158125394, 169423291, 194333347, 208742846, 238789911, 242351884,
       266971822, 295413699, 319385148},
      {16410909, 39682429, 55103279, 60931720, 87006904, 116506179, 131512017,
       152932897, 169641436, 182022480, 209277433, 237891023, 256841529,
       277007269, 290665954, 300984276},
      {18158303, 36390029, 45652312, 70783524, 98191154, 114024957, 127804916,
       159566246, 175132467, 192583600, 217919476, 237199054, 259336807,
       261799367, 289359313, 307699687},
  }};

  // Whole batch at once.
  q::NgramHistory history;
  std::vector<std::uint32_t> rows(tokens.size() * 16);
  q::HashNgramRows(c, history, tokens, rows);
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    for (std::size_t h = 0; h < 16; ++h) {
      Check(rows[i * 16 + h] == expected[i][h], "batched hash row");
    }
  }
  Check(history.prev[0] == 11 && history.prev[1] == 9707, "history tail");

  // One token at a time must agree with the batch.
  q::NgramHistory step;
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    std::array<std::uint32_t, 16> one{};
    q::HashNgramRows(c, step, std::span<const std::int32_t>(&tokens[i], 1),
                     one);
    Check(one == expected[i], "single-step hash row");
  }
}

// A BF16 table where row r holds r + i/1000 in column i; reads through the
// direct-I/O path must return exactly those values, duplicates included.
void TestTable() {
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / "qwen38_ngram_test.bin";
  const std::uint32_t dim = 160;
  const std::uint64_t rows = 300;
  const std::uint64_t offset = 4096 + 90;  // deliberately unaligned
  {
    std::ofstream out(path, std::ios::binary);
    std::vector<char> pad(offset, 0);
    out.write(pad.data(), static_cast<std::streamsize>(pad.size()));
    for (std::uint64_t r = 0; r < rows; ++r) {
      for (std::uint32_t i = 0; i < dim; ++i) {
        const float v = static_cast<float>(r) + static_cast<float>(i) / 1000.0F;
        std::uint32_t bits = 0;
        std::memcpy(&bits, &v, 4);
        const std::uint16_t bf = static_cast<std::uint16_t>(bits >> 16);
        out.write(reinterpret_cast<const char*>(&bf), 2);
      }
    }
  }
  std::string error;
  auto table = q::NgramTable::Open(path, offset, rows, dim,
                                   gufo::core::GgmlType::kBF16, &error);
  Check(table != nullptr, error.c_str());
  if (table) {
    const std::vector<std::uint32_t> ids{7, 299, 7, 0, 128, 299};
    std::vector<float> out(ids.size() * dim);
    Check(table->Read(ids, out), "table read");
    for (std::size_t j = 0; j < ids.size(); ++j) {
      for (std::uint32_t i = 0; i < dim; i += 37) {
        const float v = static_cast<float>(ids[j]) +
                        static_cast<float>(i) / 1000.0F;
        std::uint32_t bits = 0;
        std::memcpy(&bits, &v, 4);
        bits &= 0xFFFF0000U;
        float expected = 0.0F;
        std::memcpy(&expected, &bits, 4);
        Check(out[j * dim + i] == expected, "table row value");
      }
    }
    const std::vector<std::uint32_t> bad{rows};
    Check(!table->Read(bad, out), "out-of-range row rejected");
  }
  std::filesystem::remove(path);
}

}  // namespace

int main() {
  TestHash();
  TestTable();
  if (failures != 0) {
    std::fprintf(stderr, "%d failures\n", failures);
    return 1;
  }
  std::printf("PASS\n");
  return 0;
}
