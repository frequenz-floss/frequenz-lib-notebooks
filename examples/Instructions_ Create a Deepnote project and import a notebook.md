## **Instructions: Create a Deepnote project and import a Notebook**

---

### **1\. Log in to Deepnote**

Open [Deepnote](https://deepnote.com/) and log in with your account.

### **2\. Create a new project**

Select the workspace in which you want to create the new project in the side menu at the top.

Then click on the “+” under the Projects menu item and then on “New Project”

<img src="images/image1.png" width="25%">

The newly created project opens automatically:

<img src="images/image2.png" width="50%">

### **3\. Changing the Python version**

The Notebook requires Python 3.11. Since Deepnote initializes new projects with Python 3.9 by default, the version must be changed.

To do this, click on the dropdown field at the bottom left where the Python version is displayed and select Python 3.11

<img src="images/image3.png" width="25%">
<img src="images/image4.png" width="50%">

### **4\. Upload Python Notebook**

Click on the \+ in the Files tab on the left menu and select the **Upload .ipynb file option.** Then upload the Python notebook file (.ipynb).

<img src="images/image5.png" width="25%">

The uploaded notebook should be visible in the side menu under Notebooks after refreshing the page in your browser (shown as “Solar Notebook” in the image).

The original empty notebook here “Notebook 1” is no longer needed and can be deleted.

<img src="images/image6.png" width="25%">

### **5\. Install Requirements**

Click on the added Notebook in the side menu.

In order to display your data, packages must be installed before use.

To do this, run the first code cell by clicking on the blue triangle or place your cursor inside the code cell and use the keyboard shortcut "CTRL \+ Enter". You can replace the package version with the one you want.

**It is always a good idea to install the latest version as it may contain new features and/or important bug fixes.**

Example for installing [`frequenz-lib-notebooks`](https://github.com/frequenz-floss/frequenz-lib-notebooks):

<img src="images/image7.png" width="70%">

Installing the required packages may take a moment. After that, you should see a yellow box under the code that says:

“Delete this cell and move packages "frequenz-lib-notebooks"

If this is not the case, the code block must be executed again by clicking on the blue triangle again.

Click on the blue link move packages "frequenz-lib-notebooks" to “requirements.txt”

The requirements are now automatically created for all notebooks in the project, so you can create as many copies of the notebook as you want without having to perform the installation again.

### **6\. Upload other files**

If you have received a microgrids.toml file or any other file from us please upload them to your project via Files \> Upload file.

<img src="images/image8.png" width="35%">

### **7\. Enter API login data**

In order for the notebook to display your data, you must first create an integration of your Kuiper API.

To do this, click on the \+ in the **Integrations** tab on the left in the side menu and select **Create a new integration**.

Select the Environment variables option in the dropdown that opens.

<img src="images/image9.png" width="25%">

Give the integration a name of your choice and add the key value pairs listed below. 

**Key**						**Value**

REPORTING\_SERVER\_URL		grpc://reporting.api.frequenz.com:443

REPORTING\_API\_KEY			You can find your API key in Kuiper ([Anleitung](https://docs.google.com/document/u/0/d/1ePlCtr92pA1fRt2kt1PbyPs636ADk1r4PeuIOMBCM18/edit))

WEATHER\_SERVER\_URL			grpc://fz-0003.frequenz.io:50052?ssl=false

Confirm your entry by clicking on **Create integration**.

### 

### **8\. Create app**

If you want to create an app from the notebook then you can do so by clicking on the **Create app** button at the top right:

<img src="images/image10.png" width="70%">

Select the following settings (these should already be set) and confirm by clicking **Create app**:

<img src="images/image11.png" width="15%">

### **9\. Run the notebook**

Follow the specific instructions described in the notebook.


## **Instructions: Schedule regular notebook execution**

---

1\. Click the **Schedule notebook** calendar icon located at the top of the notebook, right next to the **Run notebook** button.

2\. Configure the frequency at which you want the notebook to run, then click **Save schedule**.

3\. In addition, you can also configure notifications for successful and failed runs, either via email or in Slack.

<img src="images/image12.png" width="50%">