# کارخانهٔ دادهٔ ۳۱۱ نیویورک (مرحلهٔ ۱)

از ریشهٔ پروژه، با Python 3.11 یا بالاتر:

```bash
python -m venv .venv
.venv\Scripts\activate            # Mac/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
python -m unittest discover -s tests -t .       # همهٔ تست‌ها، بدون اینترنت
data-factory run                                # کل کارخانه: fetch → validate → clean → enrich → publish
```

- فقط یک بخش: `data-factory run --stages validate,clean` — فقط دادهٔ جدید: `data-factory run --since-last-run` — بررسی تنظیمات: `data-factory check-config`
- تنظیمات در `factory.toml`. خروجی‌ها و `run_manifest.json` در `data/curated/`. پاسخ اصلی: `solution.ipynb` (Restart & Run All). دلیل تصمیم‌ها: `report.md` و `docs/decisions.md`.
- کد خروج: 0 موفق، 10 تنظیمات، 20/21 شبکه، 30–33 داده، 40 ورودی مرحلهٔ قبل موجود نیست، 1 خطای پیش‌بینی‌نشده.
