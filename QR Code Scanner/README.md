# QR Code Scanner

Standalone image/webcam QR decoder using `cv2.QRCodeDetector`. The default offline QR sample encodes `https://example.com/opencv-demo`.

## Run

```powershell
python -m pip install -r requirements.txt
python qr_scanner.py
```

Output is saved to `outputs/qr_scan_output.png`. To use webcam: `python qr_scanner.py --source 0`. Browser launching is **disabled by default**.

To enable optional browser opening, use `python qr_scanner.py --open-url`. The program will only consider plain HTTP(S) URLs and ask for explicit confirmation. QR codes can contain malicious links; inspect the domain before confirming. It will not execute commands or open non-web payloads.
