<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis minScale="1e+8" version="3.4.15-Madeira" styleCategories="AllStyleCategories" maxScale="0" hasScaleBasedVisibilityFlag="0">
  <flags>
    <Identifiable>1</Identifiable>
    <Removable>1</Removable>
    <Searchable>1</Searchable>
  </flags>
  <customproperties>
    <property key="WMSBackgroundLayer" value="false"/>
    <property key="WMSPublishDataSourceUrl" value="false"/>
    <property key="embeddedWidgets/count" value="0"/>
    <property key="identify/format" value="Value"/>
  </customproperties>
  <pipe>
    <rasterrenderer type="singlebandpseudocolor" band="1" alphaBand="-1" opacity="0.448" classificationMax="6" classificationMin="1">
      <rasterTransparency/>
      <minMaxOrigin>
        <limits>MinMax</limits>
        <extent>WholeRaster</extent>
        <statAccuracy>Estimated</statAccuracy>
        <cumulativeCutLower>0.02</cumulativeCutLower>
        <cumulativeCutUpper>0.98</cumulativeCutUpper>
        <stdDevFactor>2</stdDevFactor>
      </minMaxOrigin>
      <rastershader>
        <colorrampshader classificationMode="2" colorRampType="INTERPOLATED" clip="0">
          <colorramp type="gradient" name="[source]">
            <prop v="247,251,255,255" k="color1"/>
            <prop v="8,48,107,255" k="color2"/>
            <prop v="0" k="discrete"/>
            <prop v="gradient" k="rampType"/>
            <prop v="0.13;222,235,247,255:0.26;198,219,239,255:0.39;158,202,225,255:0.52;107,174,214,255:0.65;66,146,198,255:0.78;33,113,181,255:0.9;8,81,156,255" k="stops"/>
          </colorramp>
          <item alpha="255" color="#a79987" label="Forest" value="1"/>
          <item alpha="255" color="#213800" label="Mesic grassland" value="2"/>
          <item alpha="255" color="#00a212" label="Moist grassland" value="3"/>
          <item alpha="255" color="#ffdd01" label="Arable" value="4"/>
          <item alpha="255" color="#6b6c63" label="Pasture " value="5"/>
          <item alpha="255" color="#2072b4" label="Peatland" value="6"/>
        </colorrampshader>
      </rastershader>
    </rasterrenderer>
    <brightnesscontrast brightness="0" contrast="0"/>
    <huesaturation colorizeGreen="128" grayscaleMode="0" colorizeBlue="128" colorizeStrength="100" colorizeOn="0" saturation="0" colorizeRed="255"/>
    <rasterresampler maxOversampling="2"/>
  </pipe>
  <blendMode>0</blendMode>
</qgis>
