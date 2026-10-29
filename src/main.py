#!/usr/bin/env python3
import os
os.environ["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"
import rclpy
from rclpy.node import Node
import sys


# Import custom modules
from location_manager import LocationManager
from location_extractor import LocationExtractor
from response_generator import ResponseGenerator
from robot_navigator import RobotNavigator
from intent_detector import IntentDetector
from chat_interface import ChatInterface
from pose_publisher import PosePublisher
from response_generator import InferenceConfig

# Voice modules (optional)
try:
    from voice_input import VoiceInput
    from voice_output import VoiceOutput
    VOICE_AVAILABLE = True
except ImportError as e:
    VOICE_AVAILABLE = False
    print(f"⚠️  Voice modules not available: {e}")
    print("   Install dependencies: pip install openai-whisper sounddevice scipy pyttsx3")
    print("   Voice features will be disabled.\n")


class NavigationAssistant(Node):
    """Main navigation assistant node that coordinates all modules"""
    
    def __init__(self, enable_voice=True, voice_model="base", sensitivity="medium"):
        super().__init__('navigation_assistant')
        
        # Shared inference configuration (CPU / GPU toggle)
        self.inference_config = InferenceConfig()

        # Initialize all modules
        self.location_manager = LocationManager(self.get_logger())
        
        self.location_extractor = LocationExtractor(
            self.get_logger(),
            self.location_manager,
            inference_config=self.inference_config
        )
        
        self.response_generator = ResponseGenerator(
            self.get_logger(),
            self.location_manager,
            inference_config=self.inference_config
        )
        
        self.robot_navigator = RobotNavigator(
            self.get_logger(),
            self.location_manager,
            self
        )
        
        self.intent_detector = IntentDetector(self.get_logger())
        
        # PosePublisher — exposes /robot/pose and /robot/status for follower team
        self.pose_publisher = PosePublisher(self, self.robot_navigator)
        
        # Initialize voice modules if requested and available
        self.voice_input = None
        self.voice_output = None
        
        if enable_voice and VOICE_AVAILABLE:
            try:
                self.get_logger().info("Initializing voice modules...")
                
                # Initialize voice input (Whisper) with noise reduction
                self.voice_input = VoiceInput(
                    self.get_logger(),
                    model_size=voice_model,  # tiny, base, small, medium, large
                    noise_reduce=True  # Enable noise reduction
                )
                
                # Set sensitivity
                self.voice_input.set_sensitivity(sensitivity)
                self.get_logger().info(f"Microphone sensitivity: {sensitivity}")
                
                # Initialize voice output (TTS) with natural, Alexa-like settings
                self.voice_output = VoiceOutput(
                    self.get_logger(),
                    engine_type="piper",  # or "espeak"
                    rate=165,  # Natural speaking rate (Alexa-like)
                    volume=0.6  # Clear, audible volume
                )
                
                self.get_logger().info("✓ Voice modules initialized successfully!")
                self.get_logger().info("   Voice optimized for natural, clear speech")
                
            except Exception as e:
                self.get_logger().error(f"Failed to initialize voice modules: {e}")
                self.get_logger().warn("Continuing without voice support...")
                self.voice_input = None
                self.voice_output = None
        
        # Initialize chat interface with voice support
        self.chat_interface = ChatInterface(
            self.get_logger(),
            self.location_manager,
            self.location_extractor,
            self.response_generator,
            self.robot_navigator,
            self.intent_detector,
            self,
            voice_input=self.voice_input,
            voice_output=self.voice_output,
            pose_publisher=self.pose_publisher
        )
        
        self.get_logger().info("Navigation Assistant Ready!")
        self.get_logger().info(
            f"Available locations: {self.location_manager.get_all_locations()}"
        )
        
        if self.voice_input and self.voice_output:
            self.get_logger().info("Voice features: ENABLED ✓")
        else:
            self.get_logger().info("Voice features: DISABLED ✗")
    
    def run(self):
        """Start the chat interface"""
        self.chat_interface.chat_loop()


def print_usage():
    """Print usage information"""
    print("""
╔════════════════════════════════════════════════════════════════╗
║         TURTLEBOT NAVIGATION ASSISTANT - VOICE EDITION         ║
╠════════════════════════════════════════════════════════════════╣
║  Usage:                                                        ║
║    python3 main_voice.py [options]                            ║
║                                                                ║
║  Options:                                                      ║
║    --text-only         Disable voice features                 ║
║    --voice-model SIZE  Whisper model size (default: base)     ║
║                        Options: tiny, base, small, medium     ║
║    --sensitivity LEVEL Microphone sensitivity (default: medium)║
║                        Options: low, medium, high, very_high  ║
║    --help             Show this help message                  ║
║                                                                ║
║  Examples:                                                     ║
║    python3 main_voice.py                  # Voice enabled     ║
║    python3 main_voice.py --text-only      # Text only         ║
║    python3 main_voice.py --voice-model small                  ║
║    python3 main_voice.py --sensitivity low  # Less sensitive  ║
╚════════════════════════════════════════════════════════════════╝
""")


def main(args=None):
    # Parse command line arguments
    enable_voice = True
    voice_model = "large-v3"
    sensitivity = "medium"
    
    if '--help' in sys.argv or '-h' in sys.argv:
        print_usage()
        return
    
    if '--text-only' in sys.argv:
        enable_voice = False
        print("ℹ️  Running in TEXT-ONLY mode (voice disabled)")
    
    if '--voice-model' in sys.argv:
        try:
            idx = sys.argv.index('--voice-model')
            if idx + 1 < len(sys.argv):
                voice_model = sys.argv[idx + 1]
                valid_models = ['tiny', 'base', 'small', 'medium', 'large']
                if voice_model not in valid_models:
                    print(f"⚠️  Invalid model '{voice_model}'. Using 'base'.")
                    voice_model = 'base'
                else:
                    print(f"ℹ️  Using Whisper model: {voice_model}")
        except (IndexError, ValueError):
            print("⚠️  Invalid --voice-model argument. Using 'base'.")
    
    if '--sensitivity' in sys.argv:
        try:
            idx = sys.argv.index('--sensitivity')
            if idx + 1 < len(sys.argv):
                sensitivity = sys.argv[idx + 1]
                valid_sensitivities = ['low', 'medium', 'high', 'very_high']
                if sensitivity not in valid_sensitivities:
                    print(f"⚠️  Invalid sensitivity '{sensitivity}'. Using 'medium'.")
                    sensitivity = 'medium'
                else:
                    print(f"ℹ️  Microphone sensitivity: {sensitivity}")
        except (IndexError, ValueError):
            print("⚠️  Invalid --sensitivity argument. Using 'medium'.")
    
    # Initialize ROS2
    rclpy.init(args=args)
    
    # Create assistant
    assistant = NavigationAssistant(
        enable_voice=enable_voice,
        voice_model=voice_model,
        sensitivity=sensitivity
    )
    
    try:
        assistant.run()
    except KeyboardInterrupt:
        pass
    finally:
        assistant.robot_navigator.shutdown()
        assistant.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()