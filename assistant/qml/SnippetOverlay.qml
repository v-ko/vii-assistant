import QtQuick 2.15
import QtQuick.Window 2.15

/**
 * SnippetOverlay — one per screen.
 * Shows a translucent crosshatch overlay. User drags to select a region.
 * The selection rectangle is drawn with a red border.
 * Created/destroyed dynamically by the SnippetViewModel.
 */
Window {
    id: snippetWindow
    flags: Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Window
           | Qt.X11BypassWindowManagerHint
    color: "transparent"
    visible: true

    // Prevent WM from destroying the window (we manage lifecycle from Python)
    onClosing: function(close) { close.accepted = false }

    // Set by the loader to position on the correct screen
    property string screenName: ""

    // Grab keyboard focus so Escape works despite X11BypassWindowManagerHint
    // (the WM never gives focus to bypass windows on its own).
    Component.onCompleted: {
        requestActivate()
        overlay.forceActiveFocus()
    }

    Rectangle {
        id: overlay
        anchors.fill: parent
        color: "#28646464"  // Semi-transparent grey

        // Receive key events directly (Shortcut alone is unreliable for
        // bypass windows that the WM won't focus).
        focus: true
        Keys.onEscapePressed: snippetVM.cancel()

        // Crosshatch pattern via Canvas
        Canvas {
            anchors.fill: parent
            onPaint: {
                var ctx = getContext("2d")
                ctx.clearRect(0, 0, width, height)
                ctx.strokeStyle = "rgba(150, 150, 150, 0.3)"
                ctx.lineWidth = 1
                var spacing = 20
                // Diagonal lines (top-left to bottom-right)
                for (var i = -height; i < width; i += spacing) {
                    ctx.beginPath()
                    ctx.moveTo(i, 0)
                    ctx.lineTo(i + height, height)
                    ctx.stroke()
                }
                // Diagonal lines (top-right to bottom-left)
                for (var j = 0; j < width + height; j += spacing) {
                    ctx.beginPath()
                    ctx.moveTo(j, 0)
                    ctx.lineTo(j - height, height)
                    ctx.stroke()
                }
            }
        }

        // Selection rectangle
        Rectangle {
            id: selectionRect
            visible: mouseArea.selecting
            border.color: "red"
            border.width: 2
            color: "transparent"

            x: Math.min(mouseArea.startX, mouseArea.currentX)
            y: Math.min(mouseArea.startY, mouseArea.currentY)
            width: Math.abs(mouseArea.currentX - mouseArea.startX)
            height: Math.abs(mouseArea.currentY - mouseArea.startY)
        }

        MouseArea {
            id: mouseArea
            anchors.fill: parent
            cursorShape: Qt.CrossCursor

            property bool selecting: false
            property int startX: 0
            property int startY: 0
            property int currentX: 0
            property int currentY: 0

            onPressed: function(mouse) {
                selecting = true
                startX = mouse.x
                startY = mouse.y
                currentX = mouse.x
                currentY = mouse.y
            }

            onPositionChanged: function(mouse) {
                if (selecting) {
                    currentX = mouse.x
                    currentY = mouse.y
                }
            }

            onReleased: function(mouse) {
                selecting = false
                var rx = Math.round(Math.min(startX, mouse.x))
                var ry = Math.round(Math.min(startY, mouse.y))
                var rw = Math.round(Math.abs(mouse.x - startX))
                var rh = Math.round(Math.abs(mouse.y - startY))

                if (rw > 5 && rh > 5) {
                    console.log("SnippetOverlay: region_selected", snippetWindow.screenName, rx, ry, rw, rh)
                    snippetVM.region_selected(
                        snippetWindow.screenName, rx, ry, rw, rh
                    )
                } else {
                    // Click without drag = cancel
                    snippetVM.cancel()
                }
            }
        }
    }
}
