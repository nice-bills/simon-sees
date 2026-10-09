# Getting the phone talking to the laptop

The phone page needs the camera, and browsers only give a web page the camera over **HTTPS** (or on
`localhost`). The laptop runs the model; the phone is just a camera and a speaker. Pick one:

## Option A: Cloudflare quick tunnel (easiest, free, no account)

```bash
# laptop, terminal 1
python -m simon_sees serve            # listens on :8000

# laptop, terminal 2  (install: brew install cloudflared  |  https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/)
cloudflared tunnel --url http://localhost:8000
```

`cloudflared` prints a `https://<random-words>.trycloudflare.com` URL. Open it on the phone.

Privacy note: frames then travel phone -> Cloudflare's edge -> your laptop (TLS on both legs). Nothing is
stored or sent to any AI service, but if you want the frames to never leave your own network, use option C.

## Option B: ngrok

```bash
ngrok http 8000
```

Open the `https://….ngrok-free.app` URL on the phone. (Free ngrok shows an interstitial page once.)

## Option C: fully local, no internet at all (the "in a field" setup)

The laptop and phone only need to be on the same Wi-Fi, or the laptop on the phone's hotspot. Mobile
data can be off: hotspots work without a SIM signal.

```bash
# make a certificate the phone will trust for your laptop's LAN IP (https://github.com/FiloSottile/mkcert)
mkcert -install
mkcert 192.168.1.23 simon.local localhost         # use your laptop's IP
python -m simon_sees serve --ssl-certfile 192.168.1.23+2.pem --ssl-keyfile 192.168.1.23+2-key.pem
```

Install mkcert's root CA on the phone once (`mkcert -CAROOT` shows where it is; AirDrop/email
`rootCA.pem` to the phone. On iOS: Settings → General → VPN & Device Management → install, then
Settings → General → About → Certificate Trust Settings → enable it). Then open
`https://192.168.1.23:8000` on the phone.

Quick-and-dirty alternative: a self-signed cert and tap through the warning. Android Chrome allows it;
iOS Safari allows it but may block the WebSocket, so prefer mkcert on iPhones.

## On the phone

1. Open the URL, tap **Turn camera on**, pick the back camera (Flip), and prop the phone on a stump,
   a backpack or a cheap tripod so everyone is in frame head to toe, 3–5 m away.
2. Everyone stands in a row facing the phone and you tap **Line up**. SAM 3 finds each person and the
   team sheet fills with one photo card per player, numbered left to right. Type a name on a card, or
   (with faster-whisper on the laptop) tap 🎙 and say it; untouched cards stay "Player N".
3. Tap **Start**, turn the volume up, and walk away from the phone. Tap **Dim screen** to save battery;
   the camera keeps running.

Tips: keep the sun behind the phone, not behind the players. Leave a little space between people at
the line-up so each gets a clean photo; if a card shows two people, step apart and line up again. iOS needs the page in the foreground (Safari pauses the camera in the background).
