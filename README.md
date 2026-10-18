# 跨資產投資模擬器

以 HTML、CSS、JavaScript 和 Python Flask 製作的互動式投資回測網站。使用者在瀏覽器設定日期、預算、投入方式和比較標的；Flask 取得行情並執行回測與模型估算，再把結果交回網頁呈現。

這是教育用途的專題原型，不會連接券商或下單，也不提供個人化投資建議。

## 執行方式

需要 Python 3.10 以上和網路連線。在專案資料夾開啟 PowerShell：

    python -m venv .venv
    .\.venv\Scripts\python.exe -m pip install --upgrade pip
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    .\.venv\Scripts\python.exe app.py

看到「投資模擬器已啟動」後，在瀏覽器開啟：

    http://127.0.0.1:5000

使用完畢後，在執行程式的 PowerShell 視窗按 Ctrl+C，即可停止本機網站。

## 網站如何運作

1. templates/index.html 提供網頁內容和輸入表單。
2. static/styles.css 控制外觀，static/app.js 傳送設定並繪製結果。
3. app.py 使用 Flask 提供首頁和 /api/simulate API。
4. simulation.py 下載市場資料、計算回測指標並整理模型結果。
5. model.py 建立價格特徵、選擇模型並評估預測誤差。

瀏覽器與 Python 透過 JSON 傳遞模擬條件和計算結果。行情來自 Yahoo Finance，因此初次模擬時需要網路連線。

## 比較標的

- 台股 ETF：0050.TW
- 美股 ETF：SPY
- 比特幣：BTC-USD
- 黃金 ETF：GLD

海外資產依 USD/TWD 匯率換算成新台幣。每個標的各自使用完整預算計算，畫面是平行比較，不是混合資產的投資組合。

## 計算方式

### 歷史回測

- 一次投入：期間開始時投入全部預算。
- 定期定額：把同一筆總預算平均分到每月第一個有效交易日。
- 年化報酬：依每次投入日期和期末資產價值計算 XIRR。
- 波動度：以有效行情觀測的報酬標準差年化。ETF 使用每年 252 個交易日，比特幣使用每年 365 個日曆日。
- 最大回撤：以標的報酬指數計算，避免後續投入金額讓帳戶價值上升而低估跌幅。

週末與市場休市日不使用前一日價格補值，避免把重複報價誤當成零報酬。

### 模型估算

模型使用 5、20、60 期動能，20、60 期波動度，以及目前價格相對移動平均線的差距，預測指定有效行情觀測期間的對數報酬。

程式依時間順序保留最後 20% 資料作測試。在前段資料中比較 Random Forest 和 Ridge，再用保留測試資料計算 MAE、RMSE，並與零報酬和訓練期平均報酬基準比較。

畫面顯示的歷史情境範圍取自保留測試期實際報酬的第 10 和第 90 百分位數。這不是統計信賴區間，也不保證未來結果會落在範圍內。測試期每日預測標籤可能互相重疊，誤差數字應作描述性參考。

## 專案檔案

    app.py                                      Flask 網站與 API
    simulation.py                               行情處理和歷史回測
    model.py                                    特徵工程、選模和預測
    templates/index.html                       網頁內容與表單
    static/styles.css                          網頁樣式
    static/app.js                              表單互動、結果表格和折線圖
    requirements.txt                            Python 套件需求
    exports/跨資產投資模擬系統_專題簡報_20261019.pptx  HTML 版架構簡報

## 使用限制

- Yahoo Finance 資料可能缺漏、延遲或修訂，匯率換算也受可取得資料影響。
- 回測尚未計入手續費、稅負、滑價、ETF 配息稅務或匯兌成本。
- 模型沒有證明能持續勝過簡單基準，過去誤差不代表未來表現。
- 模型期末金額假設目前將全部預算一次投入單一標的，即使歷史回測選擇定期定額也是如此。
- 本工具供教育與方法比較使用，不是投資建議。
