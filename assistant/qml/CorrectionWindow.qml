import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

/**
 * CorrectionWindow — movable window for supervised mode review.
 * Shows agent response text and lets the supervisor approve, pass, or correct.
 */
Window {
    id: correctionWindow
    title: "Supervised Review"
    width: 600
    height: 400
    visible: correctionVM ? correctionVM.visible : false
    flags: Qt.WindowStaysOnTopHint | Qt.Tool

    // Position: bottom-center of primary screen on first show
    x: appVM.primaryScreenInfo
       ? appVM.primaryScreenInfo.x + (appVM.primaryScreenInfo.width - width) / 2
       : (Screen.width - width) / 2
    y: appVM.primaryScreenInfo
       ? appVM.primaryScreenInfo.y + appVM.primaryScreenInfo.height - height - 80
       : Screen.height - height - 80

    onVisibleChanged: {
        if (visible) {
            correctionInput.text = ""
            raise()
            requestActivate()
        }
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 10
        spacing: 8

        // ── Agent response display (reuses textDelegate style) ──
        ScrollView {
            Layout.fillWidth: true
            Layout.fillHeight: true

            TextEdit {
                id: responseDisplay
                text: correctionVM ? correctionVM.reviewText : ""
                wrapMode: Text.Wrap
                readOnly: true
                selectByMouse: true
                color: palette.text
                selectionColor: palette.highlight
                selectedTextColor: palette.highlightedText
                font.family: "monospace"
                font.pixelSize: 13
                width: parent.width
            }
        }

        // ── Correction input ────────────────────────────────────
        TextField {
            id: correctionInput
            Layout.fillWidth: true
            placeholderText: "Type correction message (for Error)…"
            Keys.onReturnPressed: {
                if (correctionInput.text.trim().length > 0) {
                    correctionVM.submitError(correctionInput.text)
                }
            }
        }

        // ── Action buttons ──────────────────────────────────────
        RowLayout {
            Layout.fillWidth: true
            spacing: 8

            Button {
                text: "✓ Correct (Alt+C)"
                Layout.fillWidth: true
                onClicked: correctionVM.submitCorrect()
            }

            Button {
                text: "→ Pass (Alt+P)"
                Layout.fillWidth: true
                onClicked: correctionVM.submitPass()
            }

            Button {
                text: "✗ Error (Alt+E)"
                Layout.fillWidth: true
                enabled: correctionInput.text.trim().length > 0
                onClicked: correctionVM.submitError(correctionInput.text)
            }
        }
    }

    // ── Keyboard shortcuts ──────────────────────────────────────
    Shortcut {
        sequence: "Alt+C"
        onActivated: correctionVM.submitCorrect()
    }
    Shortcut {
        sequence: "Alt+P"
        onActivated: correctionVM.submitPass()
    }
    Shortcut {
        sequence: "Alt+E"
        onActivated: {
            if (correctionInput.text.trim().length > 0) {
                correctionVM.submitError(correctionInput.text)
            } else {
                correctionInput.forceActiveFocus()
            }
        }
    }
    Shortcut {
        sequence: "Escape"
        onActivated: correctionVM.submitPass()
    }
}
