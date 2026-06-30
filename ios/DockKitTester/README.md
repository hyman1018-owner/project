# AutoCamTracker V1.75 Camera (iOS)

這是 AutoCamTracker 的獨立 iOS 測試工具，用來驗證自製 App 能否透過 Apple DockKit 手動控制 Insta360 Flow 2 Pro。它不是完整相機 App，也不使用 Insta360 私有 SDK。

## 已完成範圍

- DockKit accessory dock/undock 狀態監聽與硬體/韌體資訊。
- Manual Mode 狀態機：關閉 System Tracking 後輪詢驗證，成功才開放馬達控制。
- System Tracking ON/OFF 與 Flow 實體 Tracking Button 狀態監聽。
- Pan Left、Pan Right、Tilt Up、Tilt Down、STOP。
- `limits`、內建點頭動畫、相對角度與 `setAngularVelocity` 分階段能力診斷。
- `setAngularVelocity` 短測試與 `setOrientation` 回正；回正失敗時自動 STOP。
- 成功、失敗、完整 error 描述的 100 筆 UI log，可複製。
- V1.75 `TrackingCommand` JSON parser、實體相機 `zoom_factor`、dead zone、速度上限、smoothing。
- 真實 WebSocket Client、V1.75 URL 輸入、自動連線／重連、30 FPS JPEG 相機串流與 500 ms timeout STOP。
- Set Home 保存累積相對姿態 offset，Return Home 以相對 delta 回到該位置。
- predicted target / 彎道加速度時套用更積極的 smoothing 與前饋控制。
- 分頁式相機、雲台、連線與紀錄介面，支援點按對焦、變焦及直橫自動旋轉。
- DockKit System Tracking 會自動保持關閉，人物辨識與追蹤交由電腦端處理。
- App 進入背景、target lost、訊息錯誤或 timeout 時安全停止。

Apple 的軸向定義是 `Vector3D(x: pitch, y: yaw, z: roll)`；本專案依官方 sample 實作 Tilt Up 為負 pitch、Pan Left 為負 yaw。

## 需求

- macOS 與完整 Xcode（含 iOS 18 SDK；建議使用最新穩定版）。
- iPhone（iOS 18 或更新版本）。Simulator 只能看 UI，無法測 DockKit。
- Insta360 Flow 2 Pro，先透過 Insta360 App 更新韌體。
- Apple ID。自己的手機可先使用 Xcode 的 Personal Team 簽署，不需要先上架 App Store。

核心演算法可用 `swift test` 在命令列驗證；DockKit 馬達控制與硬體相容性仍必須在實體 iPhone + Flow 2 Pro 上驗收。

## 安裝到 iPhone

1. 從 Mac App Store 安裝 Xcode，第一次啟動時讓它完成 iOS platform components 安裝。
2. 開啟 `DockKitTester.xcodeproj`。
3. 在 Xcode > Settings > Accounts 登入 Apple ID。
4. 點左側藍色 DockKitTester project，選 Target `DockKitTester` > Signing & Capabilities：
   - 勾選 Automatically manage signing。
   - Team 選自己的 Personal Team 或開發團隊。
   - 若 bundle identifier 衝突，將 `com.example.DockKitTester` 改成自己的唯一值，例如 `com.yourname.DockKitTester`。
5. 用 USB 連接 iPhone，手機上按「信任這部電腦」。也可在 Xcode 配對後使用 Wi-Fi deploy。
6. iPhone 到「設定 > 隱私權與安全性 > 開發者模式」開啟 Developer Mode，依畫面重新啟動並確認。
7. Xcode 上方執行裝置選你的 iPhone，不要選 Simulator，按 `⌘R`。
8. 若 iPhone 阻擋首次啟動，到「設定 > 一般 > VPN 與裝置管理」信任對應的開發者描述檔，再開啟 App。

## Flow 2 Pro 配對與測試

1. 更新 iPhone、Insta360 App 與 Flow 2 Pro 韌體。
2. Flow 2 Pro 開機，iPhone 開啟 Bluetooth 與 Wi-Fi。
3. 依 Insta360 指示以 NFC 完成 DockKit 配對。
4. 先用 iPhone 原生相機確認 Flow 2 Pro 的 DockKit tracking 正常。
5. 把 iPhone 裝上 Flow 2 Pro，再啟動 DockKit Tester；等待狀態顯示 `Docked`。
6. 按 `Enter Manual Mode`，等待狀態顯示 `Manual Mode`；App 會自行關閉並驗證 System Tracking。
7. 依序測 Pan Left、Pan Right、Tilt Up、Tilt Down。方向鍵會持續輸出速度，每次測完立即按紅色 `STOP`。
8. 按 `Test Angular Velocity`：App 會以 yaw `+0.15 rad/s` 執行 350 ms 後停止。
9. 按 `Recenter` 或 `Test Orientation`，確認雲台回到絕對零點；不支援時 log 會顯示錯誤並執行 STOP fallback。
10. 按 `Run Capability Diagnostics`，依序驗證 limits、點頭動畫、相對角度與角速度；測試區域需保持淨空。
11. 按 `Inject Fake JSON`，確認 V1.75 資料可解碼、控制迴路有輸出，500 ms 後自動停止。
12. 複製 API Log 保存結果。判斷相容性的關鍵是 `Manual Mode ready` 與各能力測試是否成功。

## 連接 AutoCamTracker V1.75

1. Mac 的 V1.75 預設選擇 `iphone`，並自動啟動 WebSocket 與影像管線。
2. iPhone 與 Mac 使用同一 Wi-Fi，或先以 USB-C 建立 Personal Hotspot USB / USB Ethernet IP 網路。
3. App 的 `AutoCamTracker V1.75` 區會使用保存的 URL 自動連線；需要更換時可輸入 Mac 顯示的完整 URL，例如 `ws://MacBook.local:8765/ws/tracking`，再按 `Connect`。
4. V1.75 顯示 `iPhone connected` 後會接收相機影像；iOS App 會自動關閉 DockKit System Tracking，並依桌面端 `zoom_factor` 平滑調整實體相機倍率。
5. 任何無效訊息、target lost、斷線或超過 500 ms 沒有 tracking command 都會執行 STOP。

USB-C 在這一階段是「USB 上的 IP 網路」，不是自訂 raw USB protocol；因此無線和有線共用完全相同的 WebSocket URL 與 JSON 格式。單純使用 USB 安裝 App 不等於已建立 WebSocket 網路。

## 常見問題

- **一直 Not Found**：先在原生相機驗證 DockKit、重新 NFC 配對、重開 Flow 2 Pro/藍牙，並確認手機已裝到雲台上。
- **Manual Mode 失敗**：等待 accessory 顯示 Docked 後重試；保留 `setSystemTrackingEnabled(false)` 的錯誤型別與描述。
- **Tracking ON 時 STOP 顯示 skipped**：這是預期安全行為；System Tracking 擁有馬達時不應呼叫手動速度 API。
- **按下 Flow 實體 Trigger 後控制被鎖定**：實體按鍵已恢復 System Tracking，重新按 `Enter Manual Mode`。
- **API 成功但不動**：先測 `Test Angular Velocity`，再確認軸向；必要時把 `GimbalControlConfiguration.manualSpeed` 從 `0.2` 暫調到 `0.3`。
- **方向相反**：不同韌體若回報相反，調整 `GimbalVelocityCalculator` 的 yaw/pitch 正負號並記錄硬體版本。
- **Recenter 失敗**：這不阻擋核心 Pan/Tilt 驗證；App 會執行 STOP fallback。
- **Personal Team 簽署失效**：重新以 Xcode `⌘R` 安裝即可。

## 開發驗證

未安裝 Xcode 時仍可執行純 Swift 核心測試：

```sh
cd ios/DockKitTester
swift test
```

有完整 Xcode 後執行：

```sh
xcodebuild -project DockKitTester.xcodeproj \
  -scheme DockKitTester \
  -sdk iphonesimulator \
  -destination 'generic/platform=iOS Simulator' \
  CODE_SIGNING_ALLOWED=NO build
```

實際 DockKit 驗收仍必須使用 iPhone + Flow 2 Pro；Simulator 無法替代。

## 參考

- [Apple DockKit](https://developer.apple.com/documentation/dockkit)
- [Apple: Modify rotation and positioning programmatically](https://developer.apple.com/documentation/dockkit/modify-rotation-and-positioning-behavior-programmatically)
- [Apple DockKit camera sample](https://developer.apple.com/documentation/dockkit/controlling-a-dockkit-accessory-using-your-camera-app)
- [Insta360 Flow 2 Pro NFC pairing](https://onlinemanual.insta360.com/flow2pro/en-us/camera/firstuse/nfconetouchpairing)
