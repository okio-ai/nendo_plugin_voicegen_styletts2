# Nendo Plugin VoiceGen StyleTTS2

<br>
<p align="left">
    <img src="https://okio.ai/docs/assets/nendo_core_logo.png" width="350" alt="nendo core">
</p>
<br>

<p align="left">
<a href="https://okio.ai" target="_blank">
    <img src="https://img.shields.io/website/https/okio.ai" alt="Website">
</a>
<a href="https://twitter.com/okio_ai" target="_blank">
    <img src="https://img.shields.io/twitter/url/https/twitter.com/okio_ai.svg?style=social&label=Follow%20%40okio_ai" alt="Twitter">
</a>
<a href="https://discord.gg/gaZMZKzScj" target="_blank">
    <img src="https://dcbadge.vercel.app/api/server/XpkUsjwXTp?compact=true&style=flat" alt="Discord">
</a>
</p>

---

A text to speech plugin based on StyleTTS2. 

## Requirements

> [!WARNING]
> **This plugin is currently only supported on Linux.**
> We are actively working on adding support for MacOS.
> But since we are using `espeak-ng` as a backend, we are limited to Linux.

Please install the requirements for `StyleTTS2`: 

```sh
pip install git+https://github.com/resemble-ai/monotonic_align.git
sudo apt-get install espeak-ng
```


## Installation

1. [Install Nendo](https://github.com/okio-ai/nendo#installation)
2. `pip install nendo-plugin-voicegen-styletts2`

## Usage

Take a look at a basic usage example below.
For more detailed information and other plugin examples, please refer to the [documentation](https://okio.ai/docs/plugins).

```pycon
>>> from nendo import Nendo
>>> nd = Nendo(plugins=["nendo_plugin_voicegen_styletts2"])
>>> track = nd.library.add_track(file_path="path/to/file.mp3")

>>> track = nd.plugins.voicegen_styletts2(track=track)
>>> track.play()
```

    
