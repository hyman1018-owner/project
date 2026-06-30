# AutoCamTracker V1.1 更新整理

日期：2026-06-11
版本範圍：V1.0 -> V1.1
參考文件：

- `management/AutoCamTracker_Development/0609/AutoCamTracker_Development_Log.pdf`
- `management/AutoCamTracker_Development/0609/AutoCamTracker_V1_Division_Development_Log.pdf`
- `management/AutoCamTracker_Development/0610/AutoCamTracker_V1_AI_Development_Spec.pdf`
- `management/AutoCamTracker_Development/0611/video_detector_technical_report_zh.pdf`

## V1.0 原始目標

0609 規格書定義 V1 的核心方向是先完成可展示的軟體原型，不接 Sony 相機、不接 DJI 穩定器、不接 CAN。V1 重點是完成以下流程：

1. 從 MacBook webcam 或影片來源取得畫面。
2. 使用 YOLO 偵測車輛。
3. 讓使用者選擇單一車輛作為追蹤目標。
4. 計算目標 bbox 中心與畫面中心的偏移。
5. 透過 digital crop / 數位變焦模擬追蹤構圖。
6. 保留未來 V2 接實體穩定器控制的架構可能性。

0610 AI 開發規格進一步將 V1 切成五個核心模組：輸入與偵測、資料儲存、目標追蹤、構圖輸出、UI 整合。

## V1.1 修改重點

### 1. UI 重新整理

- 上方控制列改成 `Source / Tracking / Playback / View` 四個區塊。
- 四個區塊改成單列 row，使用 grid 權重平均分配，會跟著視窗大小調整。
- 移除寫死的控制區寬高，保留視窗最小尺寸，避免在不同電腦上 UI 過小。
- 所有主要按鈕改成文字可讀：`Start`、`Pause`、`Stop`、`Record`、`Auto Track`、`Reset` 等。
- 影片路徑改成短檔名顯示，避免長路徑撐爆 Source 區塊。
- 刪除 Camera index / Test Camera UI。
- 刪除 Browse Model UI，改為掃描 `code/model/` 後用 `Refresh` 更新模型清單。

### 2. 影片播放與時間軸

- 新增影片時間軸 slider，可拖曳到指定片段。
- 新增播放速度選項：`3x`、`4x`、`5x`、`6x`。
- seek 影片時會重置 detection store、target tracker、reframer 與 Ultralytics tracker state，避免不同片段沿用舊追蹤狀態。
- 影片來源關閉時可清除暫存 cache，降低從影片切回 webcam 時卡頓或狀態殘留的風險。

### 3. 追蹤與選取穩定度

- Auto Track 不再只選最大 bbox，改用穩定度分數排序。
- 穩定度分數綜合 confidence、bbox 面積、離畫面中心距離、track age。
- 點選 Before 畫面 bbox 時加入 padding，降低點到邊界時選不到的機率。
- lost 容忍時間從 15 frames 提高到 45 frames，短暫漏檢或遮擋時不會立刻 reset。
- detection threshold 從 0.25 降到 0.20，IoU 從 0.70 調到 0.65，提升快速追蹤與低信心畫面下的連續性。

### 4. 畫面顯示與操作

- Before / After 畫面支援跟隨視窗拉伸。
- View 區支援自訂 Width / Height 與 Apply Size。
- bbox label 放大到約 20 px，框線加粗，讓 track id 與 confidence 更容易看清楚。
- Before 畫面可直接點選 bbox 進行單一車輛追蹤。

### 5. 專案檔案與版本整理

- V1.1 保留 `code/` 下的程式與模型資料。
- 保留 `run_v1_app.py` 作為專案根目錄啟動入口。
- 保留 `management/AutoCamTracker_Development/` 中既有開發日誌與規格文件。
- README 改為中文簡述，聚焦程式功能、模組分工與使用方式。

## 與 V1.0 規格差異

| 項目 | V1.0 規格方向 | V1.1 實作狀態 |
| --- | --- | --- |
| UI 技術 | 原規格以 macOS C++ / Qt 為方向 | 目前 V1.1 以 Python Tkinter 實作可操作原型 |
| 輸入來源 | webcam 與影片檔測試 | 已支援 webcam、影片檔、螢幕區域 |
| 目標選擇 | 點擊縮圖或 UI 選擇單一目標 | 改為 Before 畫面直接點 bbox，並保留 Auto Track |
| 追蹤方式 | SimpleTracker / TargetSelector | 已串 YOLO track、BoT-SORT / Deep OC-SORT adapter，並加上穩定度排序 |
| 數位構圖 | digital crop 置中 | 已完成 wide / medium / close framing 與 smoothing |
| 狀態顯示 | FPS、tracking status、crop window | 已顯示 FPS、source FPS、speed、selected id、crop window |
| 影片測試 | 可讀影片檔 | 已加入時間軸、快進與 seek reset |
| 模型管理 | YOLO model path 設定 | 從 `code/model/` 掃描模型，下拉選擇 |

## 後續建議

1. 若要「車輛離開畫面後回來仍保留同一 ID」，需要加入 Vehicle ReID。
2. 建議建立 global vehicle id 層，將 tracker 的短期 id 與 ReID embedding gallery 分開管理。
3. 若要把大型 `.pt` 模型放上 GitHub，建議使用 Git LFS，避免超過 GitHub 單檔 100MB 限制。
4. 後續若轉回 Qt/C++，可沿用目前 V1.1 的模組邊界與 UI 功能作為規格。
