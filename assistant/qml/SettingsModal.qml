import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs

Popup {
    id: settingsModal
    modal: true
    focus: true
    anchors.centerIn: parent
    width: parent.width * 0.7
    height: parent.height * 0.8
    closePolicy: Popup.CloseOnEscape | Popup.CloseOnPressOutside

    onClosed: { if (settingsModalVM) settingsModalVM.hide() }

    Connections {
        target: settingsModalVM
        function onVisible_changed() {
            if (settingsModalVM.visible)
                settingsModal.open()
            else
                settingsModal.close()
        }
    }

    background: Rectangle {
        color: palette.window
        border.color: palette.mid
        border.width: 1
        radius: 8
    }

    RowLayout {
        anchors.fill: parent
        spacing: 0

        // ── Sidebar ──
        Rectangle {
            Layout.fillHeight: true
            Layout.preferredWidth: 150
            color: palette.alternateBase
            radius: 8

            // Square off right corners
            Rectangle {
                anchors.right: parent.right
                anchors.top: parent.top
                anchors.bottom: parent.bottom
                width: parent.radius
                color: parent.color
            }

            ListView {
                id: sectionList
                anchors.fill: parent
                anchors.margins: 8
                anchors.topMargin: 16
                boundsBehavior: Flickable.StopAtBounds
                model: ["Transcription"]
                currentIndex: 0

                delegate: ItemDelegate {
                    width: sectionList.width
                    text: modelData
                    highlighted: ListView.isCurrentItem
                    onClicked: sectionList.currentIndex = index
                }
            }
        }

        // ── Content area ──
        StackLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            Layout.margins: 20
            currentIndex: sectionList.currentIndex

            // ── Transcription section ──
            ColumnLayout {
                spacing: 16

                Label {
                    text: "Transcription"
                    font.pixelSize: 18
                    font.bold: true
                    color: palette.text
                }

                // ── Input device selection ──
                Label {
                    text: "Input Device"
                    font.bold: true
                    color: palette.text
                }

                ComboBox {
                    id: inputDeviceCombo
                    Layout.fillWidth: true
                    model: settingsModalVM ? settingsModalVM.inputDeviceNames : ["System Default"]
                    currentIndex: settingsModalVM ? settingsModalVM.selectedInputDeviceIndex : 0
                    onActivated: function(index) {
                        if (settingsModalVM) settingsModalVM.setInputDevice(index)
                    }
                }

                // ── File transcription drop zone ──
                Rectangle {
                    Layout.fillWidth: true
                    implicitHeight: dropContent.implicitHeight + 24
                    radius: 6
                    color: dropArea.containsDrag ? Qt.darker(palette.alternateBase, 1.1) : palette.alternateBase
                    border.color: dropArea.containsDrag ? palette.highlight : palette.mid
                    border.width: dropArea.containsDrag ? 2 : 1
                    border.pixelAligned: true

                    DropArea {
                        id: dropArea
                        anchors.fill: parent
                        keys: ["text/uri-list"]
                        onDropped: function(drop) {
                            if (drop.hasUrls && drop.urls.length > 0) {
                                var path = drop.urls[0].toString()
                                if (path.startsWith("file://")) path = path.substring(7)
                                settingsModalVM.transcribeFile(path)
                            }
                        }
                    }

                    RowLayout {
                        id: dropContent
                        anchors.fill: parent
                        anchors.margins: 12
                        spacing: 12

                        Label {
                            text: "Drop an audio file here to transcribe (result copied to clipboard)"
                            wrapMode: Text.WordWrap
                            Layout.fillWidth: true
                            color: palette.text
                        }

                        Button {
                            text: "Upload File"
                            enabled: settingsModalVM ? !settingsModalVM.sampleFileInProgress : true
                            onClicked: fileDialog.open()
                        }
                    }
                }

                RowLayout {
                    spacing: 12

                    Button {
                        text: "Open Recordings Folder"
                        onClicked: settingsModalVM.openRecordingsFolder()
                    }

                    ProgressBar {
                        Layout.fillWidth: true
                        visible: settingsModalVM ? settingsModalVM.sampleFileInProgress : false
                        value: settingsModalVM ? settingsModalVM.sampleFileProgress : 0
                        from: 0.0
                        to: 1.0
                    }

                    Label {
                        visible: settingsModalVM ? settingsModalVM.sampleFileInProgress : false
                        text: settingsModalVM ? Math.round(settingsModalVM.sampleFileProgress * 100) + "%" : ""
                        color: palette.text
                    }
                }

                Item { Layout.fillHeight: true }
            }
        }
    }

    FileDialog {
        id: fileDialog
        title: "Select audio file"
        nameFilters: ["Audio files (*.wav *.flac *.mp3 *.ogg *.m4a)", "All files (*)"]
        onAccepted: {
            var path = selectedFile.toString()
            if (path.startsWith("file://")) {
                path = path.substring(7)
            }
            settingsModalVM.transcribeFile(path)
        }
    }
}
