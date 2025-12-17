"""Audio mode configurations for different processing modes"""
from typing import Dict, List

class AudioModeConfig:
    """
    Configuration for different audio processing modes
    """
    
    MODES: Dict[str, Dict] = {
        "stadium_atmosphere": {
            "name": "Stadium Atmosphere",
            "description": "Remove commentary, keep crowd and game sounds",
            "remove_prompts": ["sports commentator", "play-by-play announcer", "human speech"],
            "keep_prompts": ["crowd cheering", "whistle", "ball impact", "sneaker squeak"],
            "operation": "remove"  # Remove the specified prompts
        },
        "commentary_only": {
            "name": "Commentary Only",
            "description": "Extract only commentary track",
            "isolate_prompts": ["sports commentator", "play-by-play announcer", "human speech"],
            "operation": "isolate"  # Keep only the specified prompts
        },
        "crowd_only": {
            "name": "Crowd Only",
            "description": "Extract only crowd reactions",
            "isolate_prompts": ["crowd cheering", "crowd chanting", "crowd booing", "crowd noise"],
            "operation": "isolate"
        },
        "game_sounds_only": {
            "name": "Game Sounds Only",
            "description": "Extract only game sounds (ball, whistle, equipment)",
            "isolate_prompts": ["ball impact", "whistle", "sneaker squeak", "equipment sound"],
            "operation": "isolate"
        },
        "referee_only": {
            "name": "Referee Only",
            "description": "Extract only referee whistle and announcements",
            "isolate_prompts": ["whistle", "referee voice"],
            "operation": "isolate"
        },
        "pure_game": {
            "name": "Pure Game",
            "description": "Remove commentary and crowd, keep only game sounds",
            "remove_prompts": ["sports commentator", "crowd cheering", "crowd noise"],
            "keep_prompts": ["ball impact", "whistle", "sneaker squeak"],
            "operation": "remove"
        },
        "original": {
            "name": "Original Audio",
            "description": "Keep original audio unchanged",
            "operation": "none"  # No processing
        }
    }
    
    @classmethod
    def get_mode_config(cls, mode: str) -> Dict:
        """Get configuration for a specific mode"""
        if mode not in cls.MODES:
            raise ValueError(f"Unknown mode: {mode}. Available modes: {list(cls.MODES.keys())}")
        return cls.MODES[mode]
    
    @classmethod
    def list_modes(cls) -> List[str]:
        """List all available modes"""
        return list(cls.MODES.keys())
    
    @classmethod
    def get_mode_info(cls, mode: str) -> str:
        """Get human-readable info about a mode"""
        config = cls.get_mode_config(mode)
        return f"{config['name']}: {config['description']}"
