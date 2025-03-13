# default-of-credit-card-clients-mlops

[![check.yml](https://github.com/hsiangjenli/default-of-credit-card-clients-mlops/actions/workflows/check.yml/badge.svg)](https://github.com/hsiangjenli/default-of-credit-card-clients-mlops/actions/workflows/check.yml)
[![publish.yml](https://github.com/hsiangjenli/default-of-credit-card-clients-mlops/actions/workflows/publish.yml/badge.svg)](https://github.com/hsiangjenli/default-of-credit-card-clients-mlops/actions/workflows/publish.yml)
[![Documentation](https://img.shields.io/badge/documentation-available-brightgreen.svg)](https://hsiangjenli.github.io/default-of-credit-card-clients-mlops/)
[![License](https://img.shields.io/github/license/hsiangjenli/default-of-credit-card-clients-mlops)](https://github.com/hsiangjenli/default-of-credit-card-clients-mlops/blob/main/LICENCE.txt)
[![Release](https://img.shields.io/github/v/release/hsiangjenli/default-of-credit-card-clients-mlops)](https://github.com/hsiangjenli/default-of-credit-card-clients-mlops/releases)

TODO.

# Installation

Use the package manager [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

# Usage

```bash
uv run default-of-credit-card-clients-mlops
```


# Data

- **ID**: 每位客戶的識別碼
- **LIMIT\_BAL**: 核發信用額度（新台幣），包括個人及家庭/附卡信用額度
- **SEX**: 性別（1=男性，2=女性）
- **EDUCATION**: 教育程度（1=研究所，2=大學，3=高中，4=其他，5=未知，6=未知）
- **MARRIAGE**: 婚姻狀況（1=已婚，2=單身，3=其他）
- **AGE**: 年齡（歲）
- **PAY\_0**: 2005 年 9 月的還款狀態（-1=準時繳款，1=逾期 1 個月，2=逾期 2 個月，以此類推，8=逾期 8 個月，9=逾期 9 個月及以上）
- **PAY\_2**: 2005 年 8 月的還款狀態（同上）
- **PAY\_3**: 2005 年 7 月的還款狀態（同上）
- **PAY\_4**: 2005 年 6 月的還款狀態（同上）
- **PAY\_5**: 2005 年 5 月的還款狀態（同上）
- **PAY\_6**: 2005 年 4 月的還款狀態（同上）
- **BILL\_AMT1**: 2005 年 9 月的帳單金額（新台幣）
- **BILL\_AMT2**: 2005 年 8 月的帳單金額（新台幣）
- **BILL\_AMT3**: 2005 年 7 月的帳單金額（新台幣）
- **BILL\_AMT4**: 2005 年 6 月的帳單金額（新台幣）
- **BILL\_AMT5**: 2005 年 5 月的帳單金額（新台幣）
- **BILL\_AMT6**: 2005 年 4 月的帳單金額（新台幣）
- **PAY\_AMT1**: 2005 年 9 月的實際還款金額（新台幣）
- **PAY\_AMT2**: 2005 年 8 月的實際還款金額（新台幣）
- **PAY\_AMT3**: 2005 年 7 月的實際還款金額（新台幣）
- **PAY\_AMT4**: 2005 年 6 月的實際還款金額（新台幣）
- **PAY\_AMT5**: 2005 年 5 月的實際還款金額（新台幣）
- **PAY\_AMT6**: 2005 年 4 月的實際還款金額（新台幣）
- **default.payment.next.month**: 是否違約（1=是，0=否）

