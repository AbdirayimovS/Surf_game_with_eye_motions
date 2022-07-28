import pyautogui
import time

def starter():
    pyautogui.keyDown('alt')
    pyautogui.press("tab")
    pyautogui.keyUp('alt')
    pyautogui.hotkey('win', 'alt', 'r')
    #pyautogui.click(1140,750, duration=1)

def motions(side):
    if side=="neutral":
        pass
    else:
        return pyautogui.press(side)

