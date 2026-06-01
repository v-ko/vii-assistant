import QtQuick

Window {
    id: overlayWindow

    // Set by the loader to position on the correct screen
    property int screenX: 0
    property int screenY: 0
    property int screenWidth: 0
    property int screenHeight: 0

    width: overlayContent.width + 32
    height: overlayContent.height + 16
    x: screenX + (screenWidth - width) / 2
    y: screenY + screenHeight * 0.03
    visible: recordingOverlayVM ? recordingOverlayVM.has_view_state : false
    color: "transparent"
    flags: Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
           | Qt.WindowTransparentForInput | Qt.WindowDoesNotAcceptFocus
           | Qt.X11BypassWindowManagerHint

    Rectangle {
        id: overlayContent
        anchors.centerIn: parent
        width: contentLoader.width + 24
        height: 40
        radius: 20
        color: "#881a1a2e"

        Loader {
            id: contentLoader
            anchors.centerIn: parent
            sourceComponent: {
                var mode = recordingOverlayVM ? recordingOverlayVM.ui_mode : ""
                if (mode === "recording")
                    return barsComponent
                if (mode === "transcribing")
                    return transcribingComponent
                return null
            }
        }
    }

    Component {
        id: barsComponent
        Row {
            spacing: 3

            Repeater {
                model: 10
                Rectangle {
                    width: 5
                    radius: 2.5
                    height: Math.max(4, recordingOverlayVM.bar_levels[index] * 28)
                    anchors.verticalCenter: parent.verticalCenter
                    color: "#44ff88"

                    Behavior on height {
                        NumberAnimation { duration: 80; easing.type: Easing.OutQuad }
                    }
                }
            }

            // Spacer before brain icon
            Item {
                width: 6
                height: 1
                visible: recordingOverlayVM.transcribing_active
            }

            // Brain badge when transcribing a chunk
            Text {
                visible: recordingOverlayVM.transcribing_active
                text: "\uD83E\uDDE0"
                font.pixelSize: 14
                opacity: 0.5
                anchors.verticalCenter: parent.verticalCenter
            }
        }
    }

    Component {
        id: transcribingComponent
        Row {
            spacing: 6
            Text {
                text: "\uD83E\uDDE0"
                font.pixelSize: 16
                anchors.verticalCenter: parent.verticalCenter
            }
            Text {
                text: "Transcribing..."
                color: "#cccccc"
                font.pixelSize: 13
                anchors.verticalCenter: parent.verticalCenter
            }
        }
    }
}
