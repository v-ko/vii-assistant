import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

// Context viewer: shows conversation items with text, images, tool calls.
// Expects `contextModel` and `settingsState` in QML context from Python.

Rectangle {
    id: contextRoot
    color: palette.base
    radius: 4
    border.color: palette.mid
    border.width: 1

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 8
        spacing: 6

        Label {
            text: "Context"
            font.bold: true
            font.pixelSize: 14
            color: palette.text
        }

        // Empty state
        Label {
            id: emptyLabel
            text: "No context items yet."
            color: palette.placeholderText
            horizontalAlignment: Text.AlignHCenter
            Layout.fillWidth: true
            Layout.fillHeight: true
            verticalAlignment: Text.AlignVCenter
            visible: contextListView.count === 0
        }

        // Scrollable context list
        ListView {
            id: contextListView
            Layout.fillWidth: true
            Layout.fillHeight: true
            clip: true
            spacing: 4
            visible: count > 0
            model: contextModel

            ScrollBar.vertical: ScrollBar {
                policy: ScrollBar.AlwaysOn
            }

            // Auto-scroll to bottom when items are added
            onCountChanged: {
                Qt.callLater(function() {
                    contextListView.positionViewAtEnd()
                })
            }

            // Auto-scroll when existing items are updated (e.g. streaming text)
            Connections {
                target: contextModel
                function onDataChanged() {
                    Qt.callLater(function() {
                        contextListView.positionViewAtEnd()
                    })
                }
            }

            delegate: Item {
                id: delegateRoot
                width: contextListView.width - 12
                x: 6
                height: delegateLoader.item ? delegateLoader.item.implicitHeight : 0

                required property int index
                required property string itemId
                required property string contentKind
                required property string text
                required property string imageB64
                required property string toolCall
                required property string requestSummary
                required property string origin

                Loader {
                    id: delegateLoader
                    anchors.left: parent.left
                    anchors.right: parent.right

                    sourceComponent: {
                        if (delegateRoot.origin === "system") return systemPromptDelegate
                        if (delegateRoot.contentKind === "image") return imageDelegate
                        if (delegateRoot.contentKind === "tool_call") return toolCallDelegate
                        return textDelegate
                    }

                    onLoaded: {
                        if (item) {
                            if ("text" in item) item.text = Qt.binding(function() { return delegateRoot.text })
                            if ("origin" in item) item.origin = Qt.binding(function() { return delegateRoot.origin })
                            if ("requestSummary" in item) item.requestSummary = Qt.binding(function() { return delegateRoot.requestSummary })
                            if ("imageB64" in item) item.imageB64 = Qt.binding(function() { return delegateRoot.imageB64 })
                            if ("toolCall" in item) item.toolCall = Qt.binding(function() { return delegateRoot.toolCall })
                        }
                    }
                }
            }
        }

        // Message input row
        RowLayout {
            Layout.fillWidth: true
            spacing: 6

            TextField {
                id: messageInput
                Layout.fillWidth: true
                placeholderText: "Type a message…"
                enabled: settingsState ? settingsState.context_updates_allowed : true
                onAccepted: submitMessage()
            }

            Button {
                id: sendButton
                text: "Send"
                enabled: settingsState ? (settingsState.context_updates_allowed && !settingsState.request_in_progress) : true
                onClicked: submitMessage()
            }
        }
    }

    function submitMessage() {
        let txt = messageInput.text.trim()
        if (txt.length > 0) {
            backend.submitMessage(txt)
            messageInput.clear()
        }
    }

    // ── Delegates ───────────────────────────────────────────────

    Component {
        id: textDelegate

        Rectangle {
            id: textItem
            property string text: ""
            property string origin: ""
            property string requestSummary: ""

            implicitHeight: textCol.implicitHeight + 12
            radius: 4
            color: origin === "user"
                ? Qt.rgba(palette.highlight.r, palette.highlight.g, palette.highlight.b, 0.15)
                : Qt.rgba(palette.mid.r, palette.mid.g, palette.mid.b, 0.1)
            border.color: Qt.rgba(palette.mid.r, palette.mid.g, palette.mid.b, 0.3)
            border.width: 1

            ColumnLayout {
                id: textCol
                anchors.fill: parent
                anchors.margins: 6
                spacing: 2

                Label {
                    text: textItem.origin || "assistant"
                    font.bold: true
                    font.pixelSize: 11
                    color: palette.dark
                    visible: textItem.origin !== ""
                }

                TextEdit {
                    text: {
                        let t = textItem.text.trim()
                        if (t.length === 0 && textItem.requestSummary) return "(…)"
                        return t
                    }
                    wrapMode: Text.Wrap
                    Layout.fillWidth: true
                    color: palette.text
                    font.italic: textItem.text.trim().length === 0 && textItem.requestSummary !== ""
                    readOnly: true
                    selectByMouse: true
                    selectionColor: palette.highlight
                    selectedTextColor: palette.highlightedText
                }
            }

            ToolTip.visible: textItem.requestSummary !== "" && textHover.hovered
            ToolTip.text: textItem.requestSummary
            HoverHandler { id: textHover }
        }
    }

    Component {
        id: imageDelegate

        Rectangle {
            id: imageItem
            property string imageB64: ""
            property string origin: ""

            implicitHeight: imgCol.implicitHeight + 12
            radius: 4
            color: Qt.rgba(palette.mid.r, palette.mid.g, palette.mid.b, 0.08)
            border.color: Qt.rgba(palette.mid.r, palette.mid.g, palette.mid.b, 0.3)
            border.width: 1

            ColumnLayout {
                id: imgCol
                anchors.fill: parent
                anchors.margins: 6
                spacing: 4

                Label {
                    text: imageItem.origin || "image"
                    font.bold: true
                    font.pixelSize: 11
                    color: palette.dark
                }

                Image {
                    id: thumbImage
                    source: imageItem.imageB64 ? "data:image/png;base64," + imageItem.imageB64 : ""
                    fillMode: Image.PreserveAspectFit
                    Layout.preferredWidth: 128
                    Layout.preferredHeight: 96
                    Layout.maximumWidth: 128
                    Layout.maximumHeight: 96
                    asynchronous: true
                    visible: imageItem.imageB64 !== ""

                    MouseArea {
                        id: thumbMouseArea
                        anchors.fill: parent
                        hoverEnabled: true
                        onEntered: {
                            if (imageItem.imageB64) {
                                imagePreview.imageSource = thumbImage.source
                                let globalPos = thumbImage.mapToGlobal(mouseX + 20, mouseY + 12)
                                imagePreview.x = globalPos.x
                                imagePreview.y = globalPos.y
                                imagePreview.visible = true
                            }
                        }
                        onExited: {
                            imagePreview.visible = false
                        }
                    }
                }

                Label {
                    text: "(invalid image)"
                    visible: imageItem.imageB64 === ""
                    color: palette.placeholderText
                    font.italic: true
                }
            }
        }
    }

    // ── Full-size image preview popup ────────────────────────
    Window {
        id: imagePreview
        visible: false
        flags: Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool | Qt.WindowTransparentForInput
        color: "transparent"
        width: previewImage.implicitWidth + 12
        height: previewImage.implicitHeight + 12

        property alias imageSource: previewImage.source

        Rectangle {
            anchors.fill: parent
            color: palette.window
            border.color: palette.mid
            border.width: 1
            radius: 4

            Image {
                id: previewImage
                anchors.centerIn: parent
                fillMode: Image.PreserveAspectFit
                // Cap at 90% of screen size
                sourceSize.width: Screen.width * 0.9
                sourceSize.height: Screen.height * 0.9
                asynchronous: true
            }
        }
    }

    Component {
        id: toolCallDelegate

        Rectangle {
            id: toolCallItem
            property string toolCall: ""
            property string origin: ""

            implicitHeight: tcCol.implicitHeight + 12
            radius: 4
            color: Qt.rgba(palette.mid.r, palette.mid.g, palette.mid.b, 0.12)
            border.color: Qt.rgba(palette.mid.r, palette.mid.g, palette.mid.b, 0.3)
            border.width: 1

            ColumnLayout {
                id: tcCol
                anchors.fill: parent
                anchors.margins: 6
                spacing: 2

                Label {
                    text: "tool_call"
                    font.bold: true
                    font.pixelSize: 11
                    color: palette.placeholderText
                }

                TextEdit {
                    text: toolCallItem.toolCall.trim() || "(…)"
                    wrapMode: Text.Wrap
                    Layout.fillWidth: true
                    color: palette.text
                    font.family: "monospace"
                    font.italic: toolCallItem.toolCall.trim().length === 0
                    readOnly: true
                    selectByMouse: true
                    selectionColor: palette.highlight
                    selectedTextColor: palette.highlightedText
                }
            }
        }
    }

    Component {
        id: systemPromptDelegate

        Rectangle {
            id: spItem
            property string text: ""

            property bool expanded: false

            implicitHeight: spCol.implicitHeight + 8
            radius: 4
            color: "transparent"

            ColumnLayout {
                id: spCol
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.margins: 4
                spacing: 2

                // Clickable header
                RowLayout {
                    spacing: 4

                    Label {
                        text: spItem.expanded ? "▼" : "▶"
                        font.pixelSize: 11
                        color: palette.dark
                    }

                    Label {
                        text: "System Prompt"
                        font.bold: true
                        font.pixelSize: 11
                        color: palette.dark
                    }

                    Item { Layout.fillWidth: true }
                }

                // Make header row clickable
                MouseArea {
                    anchors.top: parent.top
                    anchors.left: parent.left
                    anchors.right: parent.right
                    height: 24
                    cursorShape: Qt.PointingHandCursor
                    onClicked: spItem.expanded = !spItem.expanded
                }

                // Collapsible content
                TextEdit {
                    text: spItem.text.trim() || "(no system prompt)"
                    wrapMode: Text.Wrap
                    Layout.fillWidth: true
                    visible: spItem.expanded
                    color: palette.text
                    font.pixelSize: 11
                    leftPadding: 8
                    readOnly: true
                    selectByMouse: true
                    selectionColor: palette.highlight
                    selectedTextColor: palette.highlightedText

                    Rectangle {
                        anchors.left: parent.left
                        anchors.top: parent.top
                        anchors.bottom: parent.bottom
                        width: 2
                        color: palette.mid
                    }
                }
            }
        }
    }
}
