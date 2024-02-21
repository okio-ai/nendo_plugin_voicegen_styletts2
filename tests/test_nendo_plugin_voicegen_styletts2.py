from nendo import Nendo, NendoConfig, NendoTrack
import unittest

nd = Nendo(
    config=NendoConfig(
        log_level="INFO",
        plugins=["nendo_plugin_voicegen_styletts2"],
    ),
)


class VoicegenStyletts2Tests(unittest.TestCase):
    def test_run_voicegen_styletts2(self):
        nd.library.reset(force=True)
        track = nd.plugins.voicegen_styletts2(text="Hello world, finally I can talk!")
        self.assertIsNotNone(track.signal)
        self.assertIsNotNone(track.sr)

    def test_run_voicegen_styletts2_voice_cloning(self):
        nd.library.reset(force=True)
        voice_reference = nd.library.add_track(file_path="tests/assets/test.mp3")
        track = nd.plugins.voicegen_styletts2(
            text="Hello world, finally I can talk!",
            track=voice_reference,
        )
        self.assertIsNotNone(track.signal)
        self.assertIsNotNone(track.sr)


if __name__ == "__main__":
    unittest.main()
