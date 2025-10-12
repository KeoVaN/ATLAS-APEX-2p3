# Katkı Rehberi

## Branch isimleri
- `feat/<kısa-ad>` yeni özellikler
- `fix/<kısa-ad>` düzeltmeler
- `chore/<kısa-ad>` bakım/ara işler

## Commit mesajları
- Conventional Commits: `feat:`, `fix:`, `docs:`, `test:`, `chore:`

## PR kuralları
- En az 1 CODEOWNER onayı
- CI: tüm kontroller yeşil
- Açıklamaya: **Ne/Neden/Nasıl test ettim**

## Bütçe/Sözleşmeler (ATLAS APEX)
- request.* çağrıları ≤ 35
- plot sayısı ≤ 64
- max_bars_back ≤ 5000
- HTF sadece güvenli/noRepaintHTF kapısından

## Dizin Yapısı
- `src/` kaynak kod
- `docs/` dokümantasyon
- `tests/` test varlıkları ve senaryolar
- `config/` varsayılan ayarlar
- `.github/` repo iş akışları ve şablonlar
