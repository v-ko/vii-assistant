import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

// Settings panel: model/session controls, prompts, capture actions.
// Expects `settingsState`, `appVM` in QML context from Python.

Rectangle {
    id: settingsRoot
    color: palette.base
    radius: 4
    border.color: palette.mid
    border.width: 1

    property int uniformButtonHeight: 30

    RowLayout {
        anchors.fill: parent
        anchors.margins: 10
        spacing: 12

        // ── First column: Session controls & capture ────────────
        ColumnLayout {
            Layout.fillHeight: true
            Layout.preferredWidth: 1
            spacing: 8

            Button {
                id: newSessionButton
                text: "New session"
                Layout.fillWidth: true
                Layout.preferredHeight: uniformButtonHeight
                enabled: settingsState ? settingsState.session_state !== "new-session" : true
                onClicked: appVM.newSession()
                ToolTip.text: "Reset context and start a new session"
                ToolTip.visible: hovered
            }

            Button {
                text: "OCR clipboard"
                Layout.fillWidth: true
                Layout.preferredHeight: uniformButtonHeight
                enabled: settingsState ? !settingsState.assistant_working : true
                onClicked: appVM.ocrClipboard()
                ToolTip.text: "Run OCR on clipboard image"
                ToolTip.visible: hovered
            }

            // Info messages area
            ScrollView {
                Layout.fillWidth: true
                Layout.fillHeight: true

                TextArea {
                    id: infoMessages
                    readOnly: true
                    placeholderText: "Status messages"
                    wrapMode: TextArea.Wrap
                    text: settingsState ? settingsState.info_messages : ""
                    color: palette.text
                    font.pixelSize: 12

                    background: Rectangle {
                        color: palette.base
                        border.color: palette.mid
                        border.width: 1
                        radius: 2
                    }
                }
            }

            Button {
                text: "Attach screen"
                Layout.fillWidth: true
                Layout.preferredHeight: uniformButtonHeight
                enabled: settingsState ? (!settingsState.assistant_working && settingsState.context_updates_allowed) : true
                onClicked: appVM.attachScreen()
                ToolTip.text: "Capture watched screen and add to context"
                ToolTip.visible: hovered
            }

            Button {
                text: "Attach clipboard"
                Layout.fillWidth: true
                Layout.preferredHeight: uniformButtonHeight
                enabled: settingsState ? (!settingsState.assistant_working && settingsState.context_updates_allowed) : true
                onClicked: appVM.attachClipboard()
                ToolTip.text: "Add clipboard image to context"
                ToolTip.visible: hovered
            }

            // Server health status
            Label {
                id: healthLabel
                Layout.fillWidth: true
                Layout.preferredHeight: uniformButtonHeight
                verticalAlignment: Text.AlignVCenter

                property bool connected: false
                text: connected ? "Server " + appVM.serverHost + " Connected" : "Server " + appVM.serverHost + " Disconnected"
                color: connected ? "#4CAF50" : "#f44336"

                Connections {
                    target: appVM
                    function onHealth_check_done(conn, modelState, modelKey) {
                        healthLabel.connected = conn
                    }
                }
            }

            Item { Layout.fillHeight: true }
        }

        // ── Second column: Model, screen, open folders ──────────
        ColumnLayout {
            Layout.fillHeight: true
            Layout.preferredWidth: 1
            spacing: 8

            Button {
                text: "\u2699 Settings"
                Layout.fillWidth: true
                Layout.preferredHeight: uniformButtonHeight
                onClicked: settingsModalVM.show()
            }

            Button {
                text: "Open sessions folder"
                Layout.fillWidth: true
                Layout.preferredHeight: uniformButtonHeight
                onClicked: appVM.openSessionsFolder()
            }

            Button {
                text: "Open app config"
                Layout.fillWidth: true
                Layout.preferredHeight: uniformButtonHeight
                onClicked: appVM.openAppConfig()
            }

            // Model selector
            Button {
                text: "Set model"
                Layout.fillWidth: true
                Layout.preferredHeight: uniformButtonHeight
                onClicked: modelDialog.open()
            }

            // Model state label
            Label {
                id: modelStateLabel
                Layout.fillWidth: true
                Layout.preferredHeight: uniformButtonHeight
                verticalAlignment: Text.AlignVCenter
                text: {
                    let state = settingsState ? settingsState.server_model_state : "unknown"
                    let key = settingsState ? settingsState.server_model_key : ""
                    if (state === "loaded" && key) return "Model: " + key
                    if (state === "loading") return "Model: loading..."
                    if (state === "unloaded") return "Model: unloaded"
                    return "Model: unknown"
                }
                color: {
                    let state = settingsState ? settingsState.server_model_state : "unknown"
                    if (state === "loaded") return "#4CAF50"
                    if (state === "loading") return "#FFA726"
                    return "#888"
                }
            }

            // Screen selector
            ComboBox {
                id: screenCombo
                Layout.fillWidth: true
                Layout.preferredHeight: uniformButtonHeight
                model: ListModel { id: screenListModel }
                textRole: "displayName"
                valueRole: "name"

                Component.onCompleted: populateScreens()

                onActivated: function(index) {
                    let item = screenListModel.get(index)
                    if (item) appVM.setScreen(item.name)
                }

                // Re-sync when Python sets screen (e.g. after config init)
                Connections {
                    target: settingsState
                    function onScreen_changed(screenName) {
                        populateScreens()
                    }
                }
            }

            // Execution mode selector
            ComboBox {
                id: modeCombo
                Layout.fillWidth: true
                Layout.preferredHeight: uniformButtonHeight
                model: ["user-approve", "auto"]
                currentIndex: settingsState ? model.indexOf(settingsState.execution_mode) : 0

                onActivated: function(index) {
                    appVM.setExecutionMode(model[index])
                }

                Connections {
                    target: settingsState
                    function onExecution_mode_changed(mode) {
                        modeCombo.currentIndex = modeCombo.model.indexOf(mode)
                    }
                }
            }

            // Max tokens per reply
            RowLayout {
                Layout.fillWidth: true
                Layout.preferredHeight: uniformButtonHeight
                spacing: 6

                Label {
                    text: "Max tokens:"
                    color: palette.text
                    Layout.alignment: Qt.AlignVCenter
                }

                SpinBox {
                    id: maxTokensSpin
                    Layout.fillWidth: true
                    from: 16
                    to: 16384
                    stepSize: 64
                    value: settingsState ? settingsState.max_new_tokens : 256
                    editable: true

                    onValueModified: {
                        if (settingsState) settingsState.max_new_tokens = value
                    }

                    Connections {
                        target: settingsState
                        function onMax_new_tokens_changed(val) {
                            if (maxTokensSpin.value !== val)
                                maxTokensSpin.value = val
                        }
                    }
                }
            }

            Item { Layout.fillHeight: true }
        }

        // ── Third column: Prompt tabs ───────────────────────────
        ColumnLayout {
            Layout.fillHeight: true
            Layout.preferredWidth: 2
            spacing: 6

            TabBar {
                id: tabBar
                Layout.fillWidth: true

                TabButton { text: "Notes" }
                TabButton { text: "System Prompt" }
                TabButton { text: "Experiments" }
            }

            StackLayout {
                Layout.fillWidth: true
                Layout.fillHeight: true
                currentIndex: tabBar.currentIndex

                // Tab 0: Notes / user query
                ScrollView {
                    TextArea {
                        id: userQueryEdit
                        placeholderText: "Describe the user request (markdown)"
                        wrapMode: TextArea.Wrap
                        text: settingsState ? settingsState.user_query_markdown : ""
                        color: palette.text

                        onTextChanged: {
                            if (settingsState && text !== settingsState.user_query_markdown) {
                                settingsState.user_query_markdown = text
                            }
                        }

                        // Update from Python side
                        Connections {
                            target: settingsState
                            function onUser_query_changed(val) {
                                if (userQueryEdit.text !== val)
                                    userQueryEdit.text = val
                            }
                        }

                        background: Rectangle {
                            color: palette.base
                            border.color: palette.mid
                            border.width: 1
                            radius: 2
                        }
                    }
                }

                // Tab 1: System prompt
                ColumnLayout {
                    spacing: 4

                    ScrollView {
                        Layout.fillWidth: true
                        Layout.fillHeight: true

                        TextArea {
                            id: systemPromptEdit
                            placeholderText: "Define the system prompt (markdown)"
                            wrapMode: TextArea.Wrap
                            text: settingsState ? settingsState.system_prompt_markdown : ""
                            color: palette.text

                            onTextChanged: {
                                if (settingsState && text !== settingsState.system_prompt_markdown) {
                                    settingsState.system_prompt_markdown = text
                                }
                            }

                            Connections {
                                target: settingsState
                                function onSystem_prompt_changed(val) {
                                    if (systemPromptEdit.text !== val)
                                        systemPromptEdit.text = val
                                }
                            }

                            background: Rectangle {
                                color: palette.base
                                border.color: palette.mid
                                border.width: 1
                                radius: 2
                            }
                        }
                    }

                    Button {
                        text: "Add tool prompt"
                        Layout.fillWidth: true
                        onClicked: appVM.addToolPrompt()
                    }
                }

                // Tab 2: Experiments
                ColumnLayout {
                    spacing: 6

                    ComboBox {
                        id: experimentConfigCombo
                        Layout.fillWidth: true
                        textRole: "name"
                        model: ListModel { id: experimentConfigModel }

                        Component.onCompleted: {
                            var configs = appVM.getExperimentConfigs()
                            experimentConfigModel.clear()
                            var selectedIdx = 0
                            for (var i = 0; i < configs.length; i++) {
                                experimentConfigModel.append(configs[i])
                                if (configs[i].selected) selectedIdx = i
                            }
                            currentIndex = selectedIdx
                        }

                        onCurrentIndexChanged: {
                            if (currentIndex >= 0 && experimentConfigModel.count > 0) {
                                var item = experimentConfigModel.get(currentIndex)
                                if (item) appVM.setExperimentConfig(item.path)
                            }
                        }
                    }

                    RowLayout {
                        spacing: 4

                        Button {
                            text: "Step"
                            onClicked: appVM.stepExperiment()
                        }
                        Button {
                            text: "Stop"
                            onClicked: appVM.stopExperiment()
                        }
                        Button {
                            text: "Open Config"
                            onClicked: appVM.openExperimentConfig()
                        }
                    }

                    Label {
                        text: "Status: idle"
                        color: palette.text
                    }

                    Item { Layout.fillHeight: true }
                }
            }
        }
    }

    // ── Model selection dialog ──────────────────────────────────
    Dialog {
        id: modelDialog
        title: "Set model"
        anchors.centerIn: Overlay.overlay
        modal: true
        standardButtons: Dialog.Ok | Dialog.Cancel
        width: 350

        ColumnLayout {
            anchors.fill: parent
            spacing: 8

            ComboBox {
                id: modelCombo
                Layout.fillWidth: true
                model: ListModel { id: modelListModel }
                textRole: "displayName"
                valueRole: "key"

                Component.onCompleted: {
                    let models = appVM.getAvailableModels()
                    for (let i = 0; i < models.length; i++) {
                        modelListModel.append(models[i])
                    }
                    // Select current model
                    if (settingsState) {
                        for (let j = 0; j < modelListModel.count; j++) {
                            if (modelListModel.get(j).key === settingsState.selected_model) {
                                modelCombo.currentIndex = j
                                break
                            }
                        }
                    }
                }
            }
        }

        onAccepted: {
            let item = modelListModel.get(modelCombo.currentIndex)
            if (item) appVM.setModel(item.key)
        }
    }

    // ── Health check timer ──────────────────────────────────────
    Timer {
        interval: 5000
        running: settingsRoot.Window.window ? settingsRoot.Window.window.visible : false
        repeat: true
        triggeredOnStart: true
        onTriggered: appVM.scheduleHealthCheck()
    }

    function populateScreens() {
        screenListModel.clear()
        let screens = appVM.getScreenList()
        for (let i = 0; i < screens.length; i++) {
            screenListModel.append(screens[i])
        }
        // Select current screen
        if (settingsState) {
            for (let j = 0; j < screenListModel.count; j++) {
                if (screenListModel.get(j).name === settingsState.screen) {
                    screenCombo.currentIndex = j
                    break
                }
            }
        }
    }
}
