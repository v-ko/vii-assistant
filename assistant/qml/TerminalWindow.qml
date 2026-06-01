import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Window {
    id: root
    width: appVM.primaryScreenInfo ? appVM.primaryScreenInfo.width * 0.9 : Screen.width * 0.9
    height: appVM.primaryScreenInfo ? appVM.primaryScreenInfo.height / 2 : Screen.height / 2
    x: appVM.primaryScreenInfo ? appVM.primaryScreenInfo.x + (appVM.primaryScreenInfo.width - width) / 2 : (Screen.width - width) / 2
    y: appVM.primaryScreenInfo ? appVM.primaryScreenInfo.y : 0
    visible: false
    color: "transparent"
    flags: Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint

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
        onActivated: appVM.hideTerminal()
    }

    // ── React to view state visibility changes ──────────────────
    Connections {
        target: terminalState
        function onVisible_changed(vis) {
            if (vis) {
                panel.y = -panel.height
                root.visible = true
                root.raise()
                root.requestActivate()
                slideDown.start()
            } else {
                slideUp.start()
            }
        }
    }
}
