import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Window {
    id: root
    width: terminalState ? terminalState.win_width : Screen.width * 0.9
    height: terminalState ? terminalState.win_height : Screen.height / 2
    x: terminalState ? terminalState.win_x : (Screen.width - width) / 2
    y: terminalState ? terminalState.win_y : 0
    visible: false
    color: "transparent"
    flags: Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint

    property bool terminalVisible: false

    // Clip so nothing is visible outside the window area
    Item {
        id: clipper
        anchors.fill: parent
        clip: true

        Rectangle {
            id: panel
            width: clipper.width
            height: clipper.height
            radius: 5
            color: palette.window
            border.color: palette.mid
            border.width: 1

            // Start fully hidden above
            y: -height

            RowLayout {
                anchors.fill: parent
                anchors.margins: 10
                spacing: 10

                // Left panel: context viewer
                ContextViewer {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                }

                // Right panel: settings
                SettingsPanel {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                }
            }

            // Settings modal overlay
            SettingsModal {
                id: settingsModalPopup
            }
        }
    }

    // ── Explicit animations ─────────────────────────────────────
    NumberAnimation {
        id: slideDown
        target: panel
        property: "y"
        to: 0
        duration: 300
        easing.type: Easing.OutCubic
    }

    NumberAnimation {
        id: slideUp
        target: panel
        property: "y"
        to: -panel.height
        duration: 300
        easing.type: Easing.OutCubic
        onFinished: {
            root.visible = false
        }
    }

    // ── Keyboard handling ───────────────────────────────────────
    Shortcut {
        sequence: "Escape"
        onActivated: root.hideTerminal()
    }

    function showTerminal() {
        panel.y = -panel.height
        root.visible = true
        root.raise()
        root.requestActivate()
        terminalVisible = true
        slideDown.start()
    }

    function hideTerminal() {
        terminalVisible = false
        slideUp.start()
    }
}
