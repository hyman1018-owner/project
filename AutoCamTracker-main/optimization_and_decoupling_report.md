# AutoCamTracker V1.61 效能優化與架構解耦報告

本文件總結了近期針對 AutoCamTracker 所進行的核心架構重構與效能優化。主要目標是在**純 CPU 環境 (無 Nvidia 顯卡)** 下，大幅提升使用者介面的流暢度，並解決因複雜運算導致的 UI 執行緒卡頓問題。同時透過 MVC 架構的引入，提升未來專案的可維護性。

## 1. 核心效能優化 (純 CPU 效能解放)

### 1.1 背景與瓶頸
在原始架構中，ReID (特徵比對) 以及追蹤邏輯都與 Tkinter 的 `after` 迴圈綁在一起。特別是在處理多車輛同時進場時，`ReIDEmbeddingExtractor` 對每個車輛截圖進行逐一 (Sequential) 推論，導致大量 CPU 時間被消耗在模型載入與等待上，引發 UI 嚴重的延遲與卡頓 (Jitter)。

### 1.2 優化方案：批次處理 (Batch Inference)
- **變動檔案**：`src/autocamtracker/tracking/reid_embedding.py`, `src/autocamtracker/tracking/feature_gallery.py`
- **實作細節**：
  - 在 `ReIDEmbeddingExtractor` 中新增了 `extract_batch()` 方法。
  - 利用 `numpy.array` 將畫面上所有需要比對的車輛截圖打包成單一的 Batch Tensor (`N x 3 x 256 x 128`)。
  - 將這一個 Batch 傳遞給 ONNX Runtime 模型進行單次推論。
- **優化成果**：大幅減少了 ONNX 引擎啟動的 Overhead。推論次數從 `N` 次降為 `1` 次。即使在純 CPU 環境下，效能也獲得了顯著的提升，且**完全沒有改變原本的特徵權重與 GID 分配數學邏輯**。

## 2. 架構解耦 (God Object 拆解與背景運算)

### 2.1 背景與瓶頸
原始的 `main.py` 是一個高達 2200 多行的「God Object (上帝物件)」。它同時負責了：
- Tkinter UI 元件的建立與排版。
- 影片讀取與 `cv2.resize` 等影像處理。
- YOLO 偵測與 BotSORT 追蹤邏輯。
- SQLite 資料庫的讀寫。
這種緊密耦合的架構不僅難以維護，更因為所有繁重的工作都在 UI 執行緒 (Main Thread) 執行，造成只要處理稍慢，整個視窗就會凍結，甚至連拖曳視窗都會有殘影。

### 2.2 重構方案：背景工作執行緒 (TrackingWorker)
- **變動檔案**：`src/autocamtracker/core/pipeline_worker.py`, `src/autocamtracker/main.py`
- **實作細節**：
  - 將原先只負責 YOLO 的 `DetectionWorker` 升級為功能更強大的 `TrackingWorker`。
  - 將整個 `PipelineProcessor.process()` (包含 BotSORT 追蹤、ReID 比對、場景切換偵測、以及 Reframer 畫面裁切) 全部搬移到 `TrackingWorker` 內部執行。
  - 解決了 SQLite 跨執行緒存取的問題，在 `VehicleIdentityStore` 與 `FeatureGallery` 的資料庫連線中加入 `check_same_thread=False`，使背景執行緒能順利寫入身份資料。
- **優化成果**：
  - AI 運算與 UI 繪圖徹底分離。背景執行緒負責算圖並將結果封裝為 `FrameData`；UI 執行緒只負責將 `FrameData` 顯示到螢幕上。
  - 徹底解決了 UI 凍結問題。

### 2.3 重構方案：MVC 模組化拆分 (UI 解耦)
- **變動檔案**：將原 `main.py` 拆分為：
  - `src/autocamtracker/main.py` (純啟動入口，少於 20 行)
  - `src/autocamtracker/ui/app.py` (主應用 Controller，繼承各個 Mixin)
  - `src/autocamtracker/ui/mixins/ui_builder.py` (純畫面建立，負責 Grid 與 Pack)
  - `src/autocamtracker/ui/mixins/commands.py` (負責所有按鈕事件、滑鼠事件的回呼函式)
  - `src/autocamtracker/ui/mixins/identity_panel.py` (負責右側 Vehicle Database 樹狀圖的管理與更新)
  - `src/autocamtracker/ui/mixins/video_pipeline.py` (負責管理 UI 與 TrackingWorker 的資料流同步與影片更新)
- **優化成果**：程式碼可讀性大幅提升，未來若要新增按鈕或修改追蹤邏輯，可直接至對應的子模組中修改，降低了改A壞B的風險。

## 3. UI 渲染引擎優化 (修復高頻閃爍與縮放抖動)

### 3.1 背景與瓶頸
在成功解耦後，由於影片每秒更新高達 30 次以上，Tkinter 使用傳統 `tk.Label` 來顯示 `ImageTk.PhotoImage` 時，因為底層佈局引擎的重新計算，導致畫面出現人眼可見的高頻閃爍 (Screen Tearing) 以及視窗因圖片細微縮放而產生的迴圈抖動 (Layout Jitter)。

### 3.2 優化方案：Canvas 雙重緩衝渲染
- **變動檔案**：`src/autocamtracker/ui/mixins/ui_builder.py`, `src/autocamtracker/ui/mixins/video_pipeline.py`, `src/autocamtracker/ui/mixins/commands.py`
- **實作細節**：
  - 棄用會觸發 Layout 重新計算的 `tk.Label`。
  - 將 "Before Detection" 與 "After Reframe" 替換為 `tk.Canvas`。
  - 每次收到新畫面時，不再重建立 UI 元件，而是使用 `canvas.itemconfig()` 針對同一個圖層進行記憶體覆寫更新。
  - 修正了畫面點擊事件 (`on_before_click`)，讓滑鼠點擊座標能精確扣除 Canvas 為了置中產生的 Padding offset。
- **優化成果**：畫面撕裂感與閃爍完全消失，視窗尺寸徹底穩定，點擊圈選功能恢復正常，帶來如原生播放器般的極致滑順體驗。
