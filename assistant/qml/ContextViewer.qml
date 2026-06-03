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

        RowLayout {
            Layout.fillWidth: true
            spacing: 6

            Label {
                text: "Context"
                font.bold: true
                font.pixelSize: 14
                color: palette.text
            }

            Item { Layout.fillWidth: true }

            Button {
                text: "main"
                font.pixelSize: 10
                implicitHeight: 22
                onClicked: appVM.fetchContextDebug("main")
            }

            Button {
                text: "localization"
                font.pixelSize: 10
                implicitHeight: 22
                onClicked: appVM.fetchContextDebug("localization")
            }
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
            boundsBehavior: Flickable.StopAtBounds
            spacing: 4
            visible: count > 0
            model: contextModel

            // Auto-scroll: stays at bottom unless user scrolls up.
            property bool followTail: true
            property real _bottomThreshold: 30  // px

            // Detect user scroll intent when movement finishes
            onMovementEnded: {
                followTail = atYEnd || (contentHeight - contentY - height < _bottomThreshold)
            }

            // Also detect scrollbar drag
            ScrollBar.vertical: ScrollBar {
                id: vScrollBar
                policy: ScrollBar.AlwaysOn
                onPressedChanged: {
                    if (!pressed) {
                        contextListView.followTail = contextListView.atYEnd ||
                            (contextListView.contentHeight - contextListView.contentY - contextListView.height < contextListView._bottomThreshold)
                    }
                }
            }

            // When content grows (new items, streaming text, delegate layout),
            // follow the tail. Safe with incremental model ops — no loop.
            onContentHeightChanged: {
                if (followTail && contentHeight > height) {
                    positionViewAtEnd()
                }
            }

            Connections {
                target: contextModel
                // Full model reset (rare: session switch)
                function onModelReset() {
                    contextListView.followTail = true
                    Qt.callLater(function() { contextListView.positionViewAtEnd() })
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
                required property string displayText
                required property string imageB64
                required property string requestSummary
                required property string origin
                required property string focusMode

                Loader {
                    id: delegateLoader
                    anchors.left: parent.left
                    anchors.right: parent.right

                    sourceComponent: {
                        if (delegateRoot.origin === "system") return systemPromptDelegate
                        if (delegateRoot.contentKind === "image") return imageDelegate
                        return textDelegate
                    }

                    onLoaded: {
                        if (item) {
                            if ("displayText" in item) item.displayText = Qt.binding(function() { return delegateRoot.displayText })
                            if ("text" in item) item.text = Qt.binding(function() { return delegateRoot.text })
                            if ("origin" in item) item.origin = Qt.binding(function() { return delegateRoot.origin })
                            if ("focusMode" in item) item.focusMode = Qt.binding(function() { return delegateRoot.focusMode })
                            if ("requestSummary" in item) item.requestSummary = Qt.binding(function() { return delegateRoot.requestSummary })
                            if ("imageB64" in item) item.imageB64 = Qt.binding(function() { return delegateRoot.imageB64 })
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
                property bool isGenerating: settingsState ? settingsState.assistant_working : false
                text: isGenerating ? "\u25A0" : "Send"
                enabled: isGenerating || (settingsState ? settingsState.context_updates_allowed : true)
                onIsGeneratingChanged: console.log("[QML] sendButton.isGenerating =", isGenerating)
                onClicked: {
                    if (isGenerating) {
                        appVM.stopAssistant()
                    } else {
                        submitMessage()
                    }
                }
            }
        }
    }

    function submitMessage() {
        let txt = messageInput.text.trim()
        appVM.submitMessage(txt)
        messageInput.clear()
    }

    // ── Delegates ───────────────────────────────────────────────

    Component {
        id: textDelegate

        Rectangle {
            id: textItem
            property string displayText: ""
            property string origin: ""
            property string focusMode: ""
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

                RowLayout {
                    spacing: 6
                    visible: textItem.origin !== ""

                    Label {
                        text: textItem.origin || "assistant"
                        font.bold: true
                        font.pixelSize: 11
                        color: palette.dark
                    }

                    Rectangle {
                        visible: textItem.focusMode !== "" && textItem.focusMode !== "main"
                        color: Qt.rgba(palette.highlight.r, palette.highlight.g, palette.highlight.b, 0.25)
                        radius: 3
                        implicitWidth: modeLabel.implicitWidth + 8
                        implicitHeight: modeLabel.implicitHeight + 2

                        Label {
                            id: modeLabel
                            anchors.centerIn: parent
                            text: textItem.focusMode
                            font.pixelSize: 10
                            color: palette.text
                        }
                    }
                }

                TextEdit {
                    text: {
                        let t = textItem.displayText.trim()
                        if (t.length === 0 && textItem.requestSummary) return "(…)"
                        return t
                    }
                    wrapMode: Text.Wrap
                    Layout.fillWidth: true
                    color: palette.text
                    font.italic: textItem.displayText.trim().length === 0 && textItem.requestSummary !== ""
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
                Item {
                    Layout.fillWidth: true
                    implicitHeight: spHeaderRow.implicitHeight

                    RowLayout {
                        id: spHeaderRow
                        anchors.fill: parent
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

                    MouseArea {
                        anchors.fill: parent
                        cursorShape: Qt.PointingHandCursor
                        onClicked: spItem.expanded = !spItem.expanded
                    }
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

    // ── Context debug modal ─────────────────────────────────────
    Popup {
        id: contextDebugPopup
        anchors.centerIn: parent
        width: contextRoot.width * 0.9
        height: contextRoot.height * 0.85
        modal: true
        closePolicy: Popup.CloseOnEscape | Popup.CloseOnPressOutside

        background: Rectangle {
            color: palette.window
            border.color: palette.mid
            border.width: 1
            radius: 8
        }

        property string title: ""

        ColumnLayout {
            anchors.fill: parent
            spacing: 6

            RowLayout {
                Layout.fillWidth: true
                spacing: 6

                Label {
                    text: "Context: " + contextDebugPopup.title
                    font.bold: true
                    font.pixelSize: 13
                    color: palette.text
                }

                Item { Layout.fillWidth: true }

                Button {
                    text: "✕"
                    flat: true
                    implicitWidth: 28
                    implicitHeight: 24
                    onClicked: contextDebugPopup.close()
                }
            }

            ScrollView {
                Layout.fillWidth: true
                Layout.fillHeight: true
                Component.onCompleted: contentItem.boundsBehavior = Flickable.StopAtBounds

                TextArea {
                    id: contextDebugText
                    readOnly: true
                    wrapMode: Text.Wrap
                    selectByMouse: true
                    font.family: "monospace"
                    font.pixelSize: 11
                    color: palette.text
                    selectionColor: palette.highlight
                    selectedTextColor: palette.highlightedText
                }
            }
        }
    }

    Connections {
        target: appVM
        function onContext_debug_ready(focusMode, promptText) {
            contextDebugPopup.title = focusMode
            contextDebugText.text = promptText
            contextDebugPopup.open()
        }
    }
}
