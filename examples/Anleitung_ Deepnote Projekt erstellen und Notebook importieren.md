## **Anleitung: Deepnote Projekt erstellen und Notebook importieren**

---

### **1\. Einloggen in Deepnote**

Öffnen Sie [Deepnote](https://deepnote.com/) und loggen Sie sich mit Ihrem Account ein.

### **2\. Neues Projekt erstellen**

Wählen Sie oben in dem Seitenmenü den Workspace, in dem Sie das neue Projekt erstellen wollen.

Klicken sie dann bei dem Menüpunkt Projects auf das “+” und dann auf “New Project”

<img src="images/image1.png" width="25%">

Das erstellte Projekt öffnet sich automatisch:

<img src="images/image2.png" width="50%">

### **3\. Ändern der verwendeten Python Version**

Das Notebook benötigt Python 3.11. Da Deepnote standardmäßig neue Projekte mit Python 3.9 initialisiert, muss die Version geändert werden. 

Dazu klicken sie unten links auf das Dropdown Feld in dem die Python Version angezeigt wird und wählen sie Python 3.11

<img src="images/image3.png" width="25%">
<img src="images/image4.png" width="50%">

### **4\. Python Notebook hochladen**

Klicken Sie im Menü Links im Reiter **Files** auf das **\+**, wählen sie die Option **Upload .ipynb file**

<img src="images/image5.png" width="25%">

Laden Sie die zugeschickte Python Notebook Datei (.ipynb) hoch.

Das hochgeladene Notebook sollte nach einer Aktualisierung der Seite in Ihrem Browser im Seitenmenü unter Notebooks zu sehen sein (im Bild als „Solar Notebook“ angezeigt)..

Das ursprüngliche leere Notebook hier “Notebook 1” wird nicht mehr benötigt und kann gelöscht werden.

<img src="images/image6.png" width="25%">

### **5\. Requirements installieren**

Klicken Sie im Seitenmenü auf das hinzugefügte Notebook.

Damit Ihre Daten angezeigt werden können, müssen vor der Verwendung noch Pakete installiert werden. 

Führen Sie dazu die erste Codezelle durch einen Klick auf das blaue Dreieck aus oder platzieren Sie Ihren Cursor innerhalb der Codezelle und verwenden Sie die Tastenkombination „STRG \+ Eingabe“. Sie können die Paketversion durch die gewünschte ersetzen. 

**Es ist immer eine gute Idee, die neueste Version zu installieren, da diese neue Funktionen und/oder wichtige Fehlerbehebungen enthalten kann.**

Beispiel zur Installation von [`frequenz-lib-notebooks`](https://github.com/frequenz-floss/frequenz-lib-notebooks):

<img src="images/image7.png" width="70%">

Die Installation der benötigten Pakete kann einen Moment dauern. Danach sollte ein gelber Kasten unter dem Code zu sehen sein in dem steht:

“Delete this cell and move packages "frequenz-lib-notebooks"

Falls dies nicht der Fall ist, muss der Code Block durch erneutes Klicken auf das blaue Dreieck erneut ausgeführt werden.

Klicken sie auf den blauen Link move packages "frequenz-lib-notebooks" to “requirements.txt”

Die Requirements werden in dem Projekt jetzt automatisch für alle Notebooks erstellt, sodass sie beliebig viele Kopien des Notebooks erstellen können, ohne die Installation erneut durchführen zu müssen.

### **6\. Microgrids.toml-Datei hochladen**

Wenn Sie eine microgrids.toml-Datei oder eine andere Datei von uns erhalten haben, laden Sie diese bitte über Files \> Upload file in Ihr Projekt.

<img src="images/image8.png" width="35%">

### **7\. API Anmeldedaten hinterlegen**

Damit das Notebook Ihre Daten anzeigen kann, müssen Sie zunächst noch eine Integration Ihrer Kuiper API anlegen.

Klicken Sie dazu im Seitenmenü links im Reiter **Integrations** auf das **\+** und wählen Sie   
**Create a new integration.**

Wählen Sie die Option Environment variables in dem sich öffnenden Dropdown.

<img src="images/image9.png" width="25%">

Geben Sie der Integration einen Namen ihrer Wahl und fügen Sie die unten aufgelisteten Key Value Pairs hinzu. 

**Key**						**Value**

REPORTING\_SERVER\_URL		grpc://reporting.api.frequenz.com:443

REPORTING\_API\_KEY			Sie finden ihren API Key in Kuiper ([Anleitung](https://docs.google.com/document/u/0/d/1ePlCtr92pA1fRt2kt1PbyPs636ADk1r4PeuIOMBCM18/edit))

WEATHER\_SERVER\_URL			grpc://fz-0003.frequenz.io:50052?ssl=false

Bestätigen Sie ihre Eingabe durch klicken auf **Create integration**

### 

### **8\. App erstellen**

Wenn Sie aus dem Notebook eine App erstellen möchten, können Sie dies tun, indem Sie oben rechts auf die Schaltfläche **Create app** klicken:

<img src="images/image10.png" width="70%">

Wählen sie folgende Einstellungen (diese sollten bereits voreingestellt sein) und bestätigen Sie mit einem Klicken auf **Create app:**

<img src="images/image11.png" width="15%">

### **9\. Führen Sie das Notebook aus**

Befolgen Sie die im Notebook beschriebenen spezifischen Anweisungen.

## **Anleitung: Planen Sie die regelmäßige Ausführung eine Notebook**

---

1\. Klicken Sie auf das Kalendersymbol **Schedule notebook**, das sich oben im Notizbuch befindet, direkt neben der Schaltfläche **Run notebook**.

2\. Konfigurieren Sie die Häufigkeit, mit der das Notizbuch ausgeführt werden soll, und klicken Sie dann auf **Save schedule**.

3\. Darüber hinaus können Sie auch Benachrichtigungen für erfolgreiche und fehlgeschlagene Ausführungen konfigurieren, entweder per E-Mail oder in Slack.

<img src="images/image12.png" width="50%">